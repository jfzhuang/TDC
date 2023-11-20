from typing import List, Optional
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (
    ForwardResults,
    ConfigType,
    OptConfigType,
    OptMultiConfig,
    OptSampleList,
    SampleList,
    add_prefix,
)
from ..utils import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
from ..losses import accuracy


@SEGMENTORS.register_module()
class Tem_Proto_v1(BaseSegmentor):
    def __init__(
        self,
        backbone: ConfigType,
        decode_head: ConfigType,
        weight_unsup: float,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        pretrained: Optional[str] = None,
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert (
                backbone.get("pretrained") is None
            ), "both backbone and segmentor set pretrained weight"
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        self._init_decode_head(decode_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.prototype_global = nn.Parameter(
            torch.FloatTensor(self.num_classes, 128, 1, 1)
        )
        self.prototype_global.data = (
            self.decode_head.conv_seg.weight.data.detach().clone()
        )

        self.logit_scale_global = nn.Parameter(torch.FloatTensor(1))
        self.logit_scale_global.data.fill_(1)
        self.logit_scale_local = nn.Parameter(torch.FloatTensor(1))
        self.logit_scale_local.data.fill_(1)

        self.weight_unsup = weight_unsup
        self.rampup_length = train_cfg["rampup_length"]
        self.iter = 0

        assert self.with_decode_head

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def encode_decode(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # print("test inputs: ", inputs.shape)
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas, self.test_cfg)

        return seg_logits

    def sigmoid_rampup(self, current, rampup_length):
        if rampup_length == 0:
            return 1.0
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def slide_inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]["img_shape"] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(
                    crop_seg_logit,
                    (
                        int(x1),
                        int(preds.shape[3] - x2),
                        int(y1),
                        int(preds.shape[2] - y2),
                    ),
                )

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get("mode", "whole") in ["slide", "whole"], (
            f'Only "slide" or "whole" test mode are supported, but got '
            f'{self.test_cfg["mode"]}.'
        )
        ori_shape = batch_img_metas[0]["ori_shape"]
        assert all(_["ori_shape"] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == "slide":
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def predict(self, inputs: Tensor, data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0],
                )
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self, inputs: Tensor, data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def gen_prototypes(self, feat, label):
        n, c, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1).contiguous().view(n, h * w, -1)
        label = label.view(n, h * w)
        prototypes_batch = []
        for i in range(n):
            classes = torch.unique(label[i, :])
            prototypes = {}
            for c in classes:
                c = c.item()
                if c == 255:
                    continue
                prototype = feat[i, label[i, :] == c, :]
                prototype = prototype.mean(0)
                prototypes[c] = prototype
            prototypes_batch.append(prototypes)
        return prototypes_batch

    def cal_prototypes_loss(self, feats, prototypes, labels):
        loss = 0.0
        n = feats.shape[0]
        if prototypes is None:
            for i in range(n):
                feat, label = feats[i : i + 1, ...], labels[i : i + 1, ...]
                weight = self.prototype_global
                weight = F.normalize(weight, dim=1)
                feat = F.normalize(feat, dim=1)
                logit = F.conv2d(feat, weight, padding=0, stride=1)
                loss += F.cross_entropy(
                    logit * self.logit_scale_global, label, ignore_index=255
                )
        else:
            for i in range(n):
                feat, label = feats[i : i + 1, ...], labels[i : i + 1, ...]
                prototype_local = []
                keys_local = list(prototypes[i].keys())
                for k in keys_local:
                    prototype_local.append(
                        prototypes[i][k].unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
                    )
                prototype_local = torch.cat(prototype_local, dim=0)
                num_global, num_local = (
                    self.prototype_global.shape[0],
                    prototype_local.shape[0],
                )

                feat = F.normalize(feat, dim=1)
                weight_local = F.normalize(prototype_local, dim=1)
                logit_local = (
                    F.conv2d(feat, weight_local, padding=0, stride=1)
                    * self.logit_scale_local
                )
                weight_global = F.normalize(self.prototype_global, dim=1)
                logit_global = (
                    F.conv2d(feat, weight_global, padding=0, stride=1)
                    * self.logit_scale_global
                )
                logit = torch.cat([logit_global, logit_local], dim=1)
                prob = F.softmax(logit, dim=1)
                prob_global, prob_local = torch.split(
                    prob, [num_global, num_local], dim=1
                )
                prob_global = list(torch.chunk(prob_global, num_global, dim=1))
                prob_local = list(torch.chunk(prob_local, num_local, dim=1))
                for j, k in enumerate(keys_local):
                    prob_global[k] = prob_global[k] + prob_local[j]
                prob_global = torch.cat(prob_global, dim=1)
                log_prob = torch.log(1e-6 + prob_global)
                loss += F.nll_loss(log_prob, label, ignore_index=255)
        loss /= n
        return loss

    def forward(
        self,
        inputs=None,
        inputs_labeled=None,
        inputs_unlabeled=None,
        inputs_unlabeled_s=None,
        data_samples: OptSampleList = None,
        mode: str = "tensor",
    ) -> ForwardResults:
        if mode == "loss":
            return self.loss(
                inputs_labeled, inputs_unlabeled, inputs_unlabeled_s, data_samples
            )
        elif mode == "predict":
            return self.predict(inputs, data_samples)
        elif mode == "tensor":
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(
                f'Invalid mode "{mode}". ' "Only supports loss, predict and tensor mode"
            )

    def loss(
        self,
        inputs_labeled,
        inputs_unlabeled,
        inputs_unlabeled_s,
        data_samples,
    ):
        img_sup = inputs_labeled
        img_unsup = inputs_unlabeled
        img_unsup_s = inputs_unlabeled_s

        gt = [data_sample.gt_sem_seg.data for data_sample in data_samples]
        gt = torch.stack(gt, dim=0)

        gt = gt.squeeze(1)
        shape = gt.shape[1:]
        num_unsup_frames = img_unsup.shape[2]

        losses = dict()

        # supervised loss
        feats_sup = self.backbone(img_sup)
        logits_sup = self.decode_head(feats_sup)
        logits_sup = F.interpolate(
            logits_sup, size=shape, mode="bilinear", align_corners=self.align_corners
        )
        loss = dict()
        loss["loss_seg"] = self.decode_head.loss_decode(
            logits_sup, gt, ignore_index=255
        )
        loss["acc_seg"] = accuracy(logits_sup, gt)
        losses.update(add_prefix(loss, "decode"))

        # unsupervised loss
        rampup = self.sigmoid_rampup(self.iter, self.rampup_length)

        img_unsup_batch = list(torch.chunk(img_unsup, num_unsup_frames, dim=2))
        for i in range(num_unsup_frames):
            img_unsup_batch[i] = img_unsup_batch[i].squeeze(2)
        img_unsup_batch = torch.cat(img_unsup_batch, dim=0)

        feats_unsup_batch = self.backbone(img_unsup_batch)
        feats_unsup_batch = self.decode_head(feats_unsup_batch, return_feat=True)

        img_unsup_s_batch = list(torch.chunk(img_unsup_s, num_unsup_frames, dim=2))
        for i in range(num_unsup_frames):
            img_unsup_s_batch[i] = img_unsup_s_batch[i].squeeze(2)
        img_unsup_s_batch = torch.cat(img_unsup_s_batch, dim=0)

        feats_unsup_s_batch = self.backbone(img_unsup_s_batch)
        feats_unsup_s_batch = self.decode_head(feats_unsup_s_batch, return_feat=True)

        with torch.no_grad():
            logits_unsup_batch = self.decode_head.cls_seg(feats_unsup_batch)
            pl_batch = torch.argmax(logits_unsup_batch, dim=1)

        feats_unsup = torch.chunk(feats_unsup_batch, num_unsup_frames, dim=0)
        feats_unsup_s = torch.chunk(feats_unsup_s_batch, num_unsup_frames, dim=0)
        pl_batch = torch.chunk(pl_batch, num_unsup_frames, dim=0)

        prototypes = []
        for i in range(num_unsup_frames):
            prototypes.append(self.gen_prototypes(feats_unsup[i], pl_batch[i]))

        loss_prototypes = 0.0
        loss_prototypes += self.cal_prototypes_loss(feats_unsup_s[0], None, pl_batch[0])

        for i in range(num_unsup_frames - 1):
            loss_prototypes += self.cal_prototypes_loss(
                feats_unsup_s[i + 1], prototypes[i], pl_batch[i + 1]
            )
        loss_bidirectional = 0.0
        loss_bidirectional += self.cal_prototypes_loss(
            feats_unsup_s[num_unsup_frames - 1], None, pl_batch[num_unsup_frames - 1]
        )
        for i in range(num_unsup_frames - 1):
            loss_bidirectional += self.cal_prototypes_loss(
                feats_unsup_s[num_unsup_frames - i - 2],
                prototypes[num_unsup_frames - i - 1],
                pl_batch[num_unsup_frames - i - 2],
            )
        loss_prototypes = (loss_prototypes + loss_bidirectional) / 2.0

        loss["loss_proto_unsup"] = loss_prototypes * rampup * self.weight_unsup
        loss["logit_scale_global"] = self.logit_scale_global
        loss["logit_scale_local"] = self.logit_scale_local

        losses.update(add_prefix(loss, "decode"))

        rampup = torch.Tensor([rampup]).to(gt.device)
        losses["rampup"] = rampup
        self.iter += 1

        return losses

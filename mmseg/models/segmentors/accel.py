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

import sys

sys.path.insert(0, "/home/zhuangjiafan/codes/Accel")
from lib.model.flownet import FlowNets
from lib.model.warpnet import warp


@SEGMENTORS.register_module()
class Accel(BaseSegmentor):
    def __init__(
        self,
        backbone_ref: ConfigType,
        backbone_update: ConfigType,
        decode_head_ref: ConfigType,
        decode_head_update: ConfigType,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        pretrained_ref: Optional[str] = None,
        pretrained_update: Optional[str] = None,
        pretrained_flownet: Optional[str] = None,
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone_ref = MODELS.build(backbone_ref)
        self.decode_head_ref = MODELS.build(decode_head_ref)
        self.backbone_update = MODELS.build(backbone_update)
        self.decode_head_update = MODELS.build(decode_head_update)
        self.align_corners = self.decode_head_ref.align_corners
        self.num_classes = self.decode_head_ref.num_classes
        self.out_channels = self.decode_head_ref.out_channels

        self.merge = nn.Conv2d(
            self.num_classes * 2, self.num_classes, kernel_size=1, stride=1, padding=0
        )
        self.flownet = FlowNets()
        self.warp = warp()

        self.pretrained_ref = pretrained_ref
        self.pretrained_update = pretrained_update
        self.pretrained_flownet = pretrained_flownet
        self.init_weights()
        self.fix_backbone()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def init_weights(self):
        weight = torch.load(self.pretrained_ref, map_location="cpu")
        weight = weight["state_dict"]
        weight_backbone, weight_decode_head = {}, {}
        for k, v in weight.items():
            if "backbone" in k:
                k = k.replace("backbone.", "")
                weight_backbone[k] = v
            elif "decode_head" in k:
                k = k.replace("decode_head.", "")
                weight_decode_head[k] = v
        self.backbone_ref.load_state_dict(weight_backbone, True)
        self.decode_head_ref.load_state_dict(weight_decode_head, True)

        weight = torch.load(self.pretrained_update, map_location="cpu")
        weight = weight["state_dict"]
        weight_backbone, weight_decode_head = {}, {}
        for k, v in weight.items():
            if "backbone" in k:
                k = k.replace("backbone.", "")
                weight_backbone[k] = v
            elif "decode_head" in k:
                k = k.replace("decode_head.", "")
                weight_decode_head[k] = v
        self.backbone_update.load_state_dict(weight_backbone, True)
        self.decode_head_update.load_state_dict(weight_decode_head, True)

        weight = torch.load(self.pretrained_flownet, map_location="cpu")
        self.flownet.load_state_dict(weight, True)

        print("pretrained weight loaded")

    def fix_backbone(self):
        for p in self.backbone_ref.parameters():
            p.requires_grad = False
        for p in self.decode_head_ref.parameters():
            p.requires_grad = False
        for p in self.decode_head_ref.conv_seg.parameters():
            p.requires_grad = True

        for p in self.backbone_update.parameters():
            p.requires_grad = False
        for p in self.decode_head_update.parameters():
            p.requires_grad = False
        for p in self.decode_head_update.conv_seg.parameters():
            p.requires_grad = True

    def extract_feat(self, inputs: Tensor) -> Tensor:
        return

    def extract_logit_ref(self, inputs: Tensor) -> Tensor:
        with torch.no_grad():
            x = self.backbone_ref(inputs)
            x = self.decode_head_ref(x, return_feat=True)
        x = self.decode_head_ref.cls_seg(x)
        return x

    def extract_logit_update(self, inputs: Tensor) -> Tensor:
        with torch.no_grad():
            x = self.backbone_update(inputs)
            x = self.decode_head_update(x, return_feat=True)
        x = self.decode_head_update.cls_seg(x)
        # print("x:", x.min().item(), x.max().item(), x.mean().item())
        return x

    def transform_img_1(self, img):
        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375]
        img = img.clone().permute(1, 2, 0).detach().cpu().numpy()
        img = img * std + mean
        img = img.astype(np.uint8)
        return img

    def transform_img_2(self, img):
        img = img.clone().permute(1, 2, 0).detach().cpu().numpy()
        img = img * 255.0
        img = img.astype(np.uint8)
        return img

    def decode_labels(self, mask):
        label_colours = [
            (128, 64, 128),
            (244, 35, 231),
            (69, 69, 69),
            (102, 102, 156),
            (190, 153, 153),
            (153, 153, 153),
            (250, 170, 29),
            (219, 219, 0),
            (106, 142, 35),
            (152, 250, 152),
            (69, 129, 180),
            (219, 19, 60),
            (255, 0, 0),
            (0, 0, 142),
            (0, 0, 69),
            (0, 60, 100),
            (0, 79, 100),
            (0, 0, 230),
            (119, 10, 32),
            (0, 0, 0),
        ]

        h, w = mask.shape
        mask[mask == 255] = 19
        color_table = np.array(label_colours, dtype=np.float32)
        out = np.take(color_table, mask, axis=0)
        out = out.astype(np.uint8)
        out = out[:, :, ::-1]
        return out

    def transform_gt(self, gt):
        gt = gt.clone().detach().cpu().numpy()
        gt = self.decode_labels(gt)
        return gt

    def encode_decode(
        self, inputs: Tensor, inputs_flow: Tensor, batch_img_metas: List[dict] = None
    ) -> Tensor:
        num_unsup_frames = inputs.shape[2]
        inputs = list(torch.chunk(inputs, num_unsup_frames, dim=2))
        inputs_flow = list(torch.chunk(inputs_flow, num_unsup_frames, dim=2))
        for i in range(num_unsup_frames):
            inputs[i] = inputs[i].squeeze(2)
            inputs_flow[i] = inputs_flow[i].squeeze(2)

        logits_ref = self.extract_logit_ref(inputs[0])
        logits_ref = F.interpolate(
            logits_ref,
            scale_factor=2,
            mode="bilinear",
            align_corners=self.align_corners,
        )

        for i in range(num_unsup_frames - 1):
            flow = self.flownet(torch.cat([inputs_flow[i + 1], inputs_flow[i]], dim=1))
            logits_ref = self.warp(logits_ref, flow)

        logits_update = self.extract_logit_update(inputs[-1])
        logits_update = F.interpolate(
            logits_update,
            scale_factor=2,
            mode="bilinear",
            align_corners=self.align_corners,
        )

        logits_merge = self.merge(torch.cat([logits_ref, logits_update], dim=1))
        logits_merge = F.interpolate(
            logits_merge,
            scale_factor=4,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return logits_merge

    def whole_inference(
        self, inputs: Tensor, inputs_flow: Tensor, batch_img_metas: List[dict]
    ) -> Tensor:
        seg_logits = self.encode_decode(inputs, inputs_flow, batch_img_metas)
        return seg_logits

    def inference(
        self, inputs: Tensor, inputs_flow: Tensor, batch_img_metas: List[dict]
    ) -> Tensor:
        assert self.test_cfg.get("mode", "whole") in ["slide", "whole"], (
            f'Only "slide" or "whole" test mode are supported, but got '
            f'{self.test_cfg["mode"]}.'
        )
        ori_shape = batch_img_metas[0]["ori_shape"]
        assert all(_["ori_shape"] == ori_shape for _ in batch_img_metas)
        seg_logit = self.whole_inference(inputs, inputs_flow, batch_img_metas)

        return seg_logit

    def predict(
        self, inputs: Tensor, inputs_flow: Tensor, data_samples: OptSampleList = None
    ) -> SampleList:
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

        seg_logits = self.inference(inputs, inputs_flow, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(
        self, inputs: Tensor, inputs_flow: Tensor, data_samples: OptSampleList = None
    ) -> Tensor:
        x = self.encode_decode(inputs, inputs_flow)
        return x

    def forward(
        self,
        inputs=None,
        inputs_flow=None,
        data_samples: OptSampleList = None,
        mode: str = "tensor",
    ) -> ForwardResults:
        if mode == "loss":
            return self.loss(inputs, inputs_flow, data_samples)
        elif mode == "predict":
            return self.predict(inputs, inputs_flow, data_samples)
        elif mode == "tensor":
            return self._forward(inputs, inputs_flow, data_samples)
        else:
            raise RuntimeError(
                f'Invalid mode "{mode}". ' "Only supports loss, predict and tensor mode"
            )

    def evaluate(self, inputs, inputs_flow):
        num_unsup_frames = inputs.shape[2]
        inputs = list(torch.chunk(inputs, num_unsup_frames, dim=2))
        inputs_flow = list(torch.chunk(inputs_flow, num_unsup_frames, dim=2))
        for i in range(num_unsup_frames):
            inputs[i] = inputs[i].squeeze(2)
            inputs_flow[i] = inputs_flow[i].squeeze(2)

        out_list = []
        logits_ref = self.extract_logit_ref(inputs[0])
        logits_ref = F.interpolate(
            logits_ref,
            scale_factor=2,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        out = F.interpolate(
            logits_ref,
            scale_factor=4,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        out = torch.argmax(out, dim=1)
        out_list.append(out)

        for i in range(num_unsup_frames - 1):
            flow = self.flownet(torch.cat([inputs_flow[i + 1], inputs_flow[i]], dim=1))
            logits_ref = self.warp(logits_ref, flow)

            logits_update = self.extract_logit_update(inputs[i + 1])
            logits_update = F.interpolate(
                logits_update,
                scale_factor=2,
                mode="bilinear",
                align_corners=self.align_corners,
            )

            logits_merge = self.merge(torch.cat([logits_ref, logits_update], dim=1))
            out = F.interpolate(
                logits_merge,
                scale_factor=4,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            out = torch.argmax(out, dim=1)
            out_list.append(out)
        return out_list

    def loss(self, inputs, inputs_flow, data_samples):
        gt = [data_sample.gt_sem_seg.data for data_sample in data_samples]
        gt = torch.stack(gt, dim=0)
        gt = gt.squeeze(1)

        logits = self.encode_decode(inputs, inputs_flow)

        losses = dict()
        loss = dict()
        loss["loss_seg"] = self.decode_head_ref.loss_decode(
            logits, gt, ignore_index=255
        )
        loss["acc_seg"] = accuracy(logits, gt)
        losses.update(add_prefix(loss, "decode"))

        return losses

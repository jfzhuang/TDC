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
from lib.model.flownet import FlowNets_DFF
from lib.model.warpnet import warp


@SEGMENTORS.register_module()
class DFF(BaseSegmentor):
    def __init__(
        self,
        backbone_ref: ConfigType,
        decode_head_ref: ConfigType,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        pretrained_ref: Optional[str] = None,
        pretrained_flownet: Optional[str] = None,
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone_ref = MODELS.build(backbone_ref)
        self.decode_head_ref = MODELS.build(decode_head_ref)
        self.align_corners = self.decode_head_ref.align_corners
        self.num_classes = self.decode_head_ref.num_classes
        self.out_channels = self.decode_head_ref.out_channels

        self.flownet = FlowNets_DFF()
        self.warp = warp()

        self.pretrained_ref = pretrained_ref
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

        weight = torch.load(self.pretrained_flownet, map_location="cpu")
        self.flownet.load_state_dict(weight, False)

        print("pretrained weight loaded")

    def fix_backbone(self):
        for p in self.backbone_ref.parameters():
            p.requires_grad = False
        for p in self.decode_head_ref.parameters():
            p.requires_grad = False

    def extract_feat(self, inputs: Tensor) -> Tensor:
        return

    def extract_logit_ref(self, inputs: Tensor) -> Tensor:
        with torch.no_grad():
            x = self.backbone_ref(inputs)
            x = self.decode_head_ref(x, return_feat=True)
        x = self.decode_head_ref.cls_seg(x)
        return x

    def encode_decode(
        self, inputs: Tensor, inputs_flow: Tensor, batch_img_metas: List[dict] = None
    ) -> Tensor:
        img_s, _ = list(torch.chunk(inputs, 2, dim=2))
        img_flow_s, img_flow_t = list(torch.chunk(inputs_flow, 2, dim=2))
        img_s, img_flow_s, img_flow_t = (
            img_s.squeeze(2),
            img_flow_s.squeeze(2),
            img_flow_t.squeeze(2),
        )

        logits_ref = self.extract_logit_ref(img_s)
        logits_ref = F.interpolate(
            logits_ref,
            scale_factor=2,
            mode="bilinear",
            align_corners=self.align_corners,
        )

        flow, scale = self.flownet(torch.cat([img_flow_t, img_flow_s], dim=1))
        logits_ref = self.warp(logits_ref, flow)
        logits_ref = logits_ref * scale

        logits_ref = F.interpolate(
            logits_ref,
            scale_factor=4,
            mode="bilinear",
            align_corners=self.align_corners,
        )

        return logits_ref

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

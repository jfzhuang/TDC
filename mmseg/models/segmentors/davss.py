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


class DMNet(nn.Module):
    def __init__(self):
        super(DMNet, self).__init__()

        self.conv1 = nn.Sequential(
            SeparableConv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SyncBatchNorm(4),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            SeparableConv2d(4, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SyncBatchNorm(8),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            SeparableConv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SyncBatchNorm(16),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=True)

        self.weight_init()

    def forward(self, feat1, feat2):
        feat1 = self.conv1(feat1)
        feat1 = self.conv2(feat1)
        feat1 = self.conv3(feat1)
        feat1 = self.conv4(feat1)

        feat2 = self.conv1(feat2)
        feat2 = self.conv2(feat2)
        feat2 = self.conv3(feat2)
        feat2 = self.conv4(feat2)

        diff = -F.cosine_similarity(feat1, feat2, dim=1)
        diff = diff * 0.5 + 0.5
        diff = torch.clamp(diff, min=0.0, max=1.0)
        diff = diff.unsqueeze(1)
        return diff

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            inplanes,
            inplanes,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=inplanes,
            bias=bias,
        )
        self.depthwise_bn = nn.SyncBatchNorm(inplanes, eps=1e-05, momentum=0.0003)
        self.depthwise_relu = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
        )
        self.pointwise_bn = nn.SyncBatchNorm(planes, eps=1e-05, momentum=0.0003)
        self.pointwise_relu = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_relu(x)
        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        x = self.pointwise_relu(x)
        return x

    def _init_weight(self):
        nn.init.normal_(self.depthwise.weight, std=0.33)
        self.depthwise_bn.weight.data.fill_(1)
        self.depthwise_bn.bias.data.zero_()
        nn.init.normal_(self.pointwise.weight, std=0.33)
        self.pointwise_bn.weight.data.fill_(1)
        self.pointwise_bn.bias.data.zero_()


class CFNet(nn.Module):
    def __init__(self, in_planes=3, n_classes=19):
        super(CFNet, self).__init__()

        self.conv1 = self.conv(in_planes, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = self.conv(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = self.conv(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3_1 = self.conv(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = self.conv(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = self.conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = self.conv(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = self.conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = self.conv(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv6_1 = self.conv(512, 512, kernel_size=3, stride=1, padding=1)

        self.deconv5 = self.deconv(512, 256)
        self.deconv4 = self.deconv(512, 128)
        self.deconv3 = self.deconv(384, 64)
        self.deconv2 = self.deconv(192, 32)

        self.predict_cc = nn.Conv2d(96, n_classes, 1, stride=1, padding=0)

    def forward(self, image):
        out_conv1 = self.conv1(image)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        out_deconv5 = self.antipad(
            self.deconv5(out_conv6),
            evenh=out_conv5.shape[2] % 2 == 0,
            evenw=out_conv5.shape[3] % 2 == 0,
        )
        concat5 = torch.cat((out_conv5, out_deconv5), 1)

        out_deconv4 = self.antipad(
            self.deconv4(concat5),
            evenh=out_conv4.shape[2] % 2 == 0,
            evenw=out_conv4.shape[3] % 2 == 0,
        )
        concat4 = torch.cat((out_conv4, out_deconv4), 1)

        out_deconv3 = self.antipad(
            self.deconv3(concat4),
            evenh=out_conv3.shape[2] % 2 == 0,
            evenw=out_conv3.shape[3] % 2 == 0,
        )
        concat3 = torch.cat((out_conv3, out_deconv3), 1)

        out_deconv2 = self.antipad(
            self.deconv2(concat3),
            evenh=out_conv2.shape[2] % 2 == 0,
            evenw=out_conv2.shape[3] % 2 == 0,
        )
        concat2 = torch.cat((out_conv2, out_deconv2), 1)

        correction_cue = self.predict_cc(concat2)

        return correction_cue

    def conv(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.SyncBatchNorm(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def deconv(self, in_planes, out_planes):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_planes, out_planes, kernel_size=4, stride=2, padding=0, bias=False
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def antipad(self, tensor, evenh=True, evenw=True, num=1):
        h = tensor.shape[2]
        w = tensor.shape[3]
        if evenh and evenw:
            tensor = tensor.narrow(2, 1, h - 2 * num)
            tensor = tensor.narrow(3, 1, w - 2 * num)
            return tensor
        elif evenh and (not evenw):
            tensor = tensor.narrow(2, 1, h - 2 * num)
            tensor = tensor.narrow(3, 1, w - 2 * num - 1)
            return tensor
        elif (not evenh) and evenw:
            tensor = tensor.narrow(2, 1, h - 2 * num - 1)
            tensor = tensor.narrow(3, 1, w - 2 * num)
            return tensor
        else:
            tensor = tensor.narrow(2, 1, h - 2 * num - 1)
            tensor = tensor.narrow(3, 1, w - 2 * num - 1)
            return tensor


@SEGMENTORS.register_module()
class SCNet_DMNet(BaseSegmentor):
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

        self.dmnet = DMNet()
        self.flownet = FlowNets()
        self.warp = warp()

        self.pretrained_ref = pretrained_ref
        self.pretrained_flownet = pretrained_flownet
        self.init_weights()
        self.fix_backbone()
        self.fix_flownet()

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

    def fix_flownet(self):
        for param in self.flownet.parameters():
            param.requires_grad = False

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
        return

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
        img_1_s, img_2_s, img_3_s = list(torch.chunk(inputs, 3, dim=2))
        img_1_f, img_2_f, img_3_f = list(torch.chunk(inputs_flow, 3, dim=2))
        img_1_s, img_2_s, img_3_s = (
            img_1_s.squeeze(2),
            img_2_s.squeeze(2),
            img_3_s.squeeze(2),
        )
        img_1_f, img_2_f, img_3_f = (
            img_1_f.squeeze(2),
            img_2_f.squeeze(2),
            img_3_f.squeeze(2),
        )

        with torch.no_grad():
            pred_1 = self.extract_logit_ref(img_1_s)
            pred_1 = F.interpolate(
                pred_1, scale_factor=2, mode="bilinear", align_corners=False
            )

            pred_2 = self.extract_logit_ref(img_2_s)
            pred_2 = F.interpolate(
                pred_2, scale_factor=8, mode="bilinear", align_corners=False
            )
            out_2 = torch.argmax(pred_2, dim=1)

            pred_3 = self.extract_logit_ref(img_3_s)
            pred_3 = F.interpolate(
                pred_3, scale_factor=8, mode="bilinear", align_corners=False
            )
            out_3 = torch.argmax(pred_3, dim=1)

            img_1_s = F.interpolate(
                img_1_s, scale_factor=0.25, mode="bilinear", align_corners=False
            )
            img_2_s = F.interpolate(
                img_2_s, scale_factor=0.25, mode="bilinear", align_corners=False
            )
            img_3_s = F.interpolate(
                img_3_s, scale_factor=0.25, mode="bilinear", align_corners=False
            )

            flow_12 = self.flownet(torch.cat([img_2_f, img_1_f], dim=1))
            flow_23 = self.flownet(torch.cat([img_3_f, img_2_f], dim=1))

            img_2_warp = self.warp(img_1_s, flow_12)
            img_3_warp = self.warp(img_2_warp, flow_23)

        with torch.no_grad():
            pred_2_warp = self.warp(pred_1, flow_12)
            pred_3_warp = self.warp(pred_2_warp, flow_23)

            pred_2_warp = F.interpolate(
                pred_2_warp, scale_factor=4, mode="bilinear", align_corners=False
            )
            out_2_warp = torch.argmax(pred_2_warp, dim=1, keepdims=True)
            label_2 = (out_2_warp != out_2.unsqueeze(1)).float().detach()
            pred_3_warp = F.interpolate(
                pred_3_warp, scale_factor=4, mode="bilinear", align_corners=False
            )
            out_3_warp = torch.argmax(pred_3_warp, dim=1, keepdims=True)
            label_3 = (out_3_warp != out_3.unsqueeze(1)).float().detach()

        loss_dmnet = 0.0

        dm_2 = self.dmnet(img_2_warp, img_2_s)
        dm_2 = F.interpolate(dm_2, scale_factor=4, mode="bilinear", align_corners=False)
        loss_dmnet += F.binary_cross_entropy(dm_2, label_2)

        dm_3 = self.dmnet(img_3_warp, img_3_s)
        dm_3 = F.interpolate(dm_3, scale_factor=4, mode="bilinear", align_corners=False)
        loss_dmnet += F.binary_cross_entropy(dm_3, label_3)
        loss_dmnet /= 2

        losses = dict()
        loss = dict()
        loss["loss_dmnet"] = loss_dmnet
        losses.update(add_prefix(loss, "decode"))

        return losses


@SEGMENTORS.register_module()
class SCNet(SCNet_DMNet):
    def __init__(self, pretrained_dmnet: Optional[str] = None, **kwargs):
        self.pretrained_dmnet = pretrained_dmnet

        super().__init__(**kwargs)
        self.cfnet = CFNet(n_classes=self.num_classes)

        self.fix_dmnet()
        self.train_flownet()

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
        self.flownet.load_state_dict(weight, True)

        weight = torch.load(self.pretrained_dmnet, map_location="cpu")
        weight = weight["state_dict"]
        weight_dmnet = {}
        for k, v in weight.items():
            if "dmnet" in k:
                k = k.replace("dmnet.", "")
                weight_dmnet[k] = v
        self.dmnet.load_state_dict(weight_dmnet, True)

        print("pretrained weight loaded")

    def fix_dmnet(self):
        for p in self.dmnet.parameters():
            p.requires_grad = False

    def train_flownet(self):
        for p in self.flownet.parameters():
            p.requires_grad = True

    def encode_decode(
        self, inputs: Tensor, inputs_flow: Tensor, batch_img_metas: List[dict] = None
    ) -> Tensor:
        return

    def loss(self, inputs, inputs_flow, data_samples):
        gt = [data_sample.gt_sem_seg.data for data_sample in data_samples]
        gt = torch.stack(gt, dim=0)
        gt = gt.squeeze(1)

        img_1_s, img_2_s, img_3_s = list(torch.chunk(inputs, 3, dim=2))
        img_1_f, img_2_f, img_3_f = list(torch.chunk(inputs_flow, 3, dim=2))
        img_1_s, img_2_s, img_3_s = (
            img_1_s.squeeze(2),
            img_2_s.squeeze(2),
            img_3_s.squeeze(2),
        )
        img_1_f, img_2_f, img_3_f = (
            img_1_f.squeeze(2),
            img_2_f.squeeze(2),
            img_3_f.squeeze(2),
        )

        with torch.no_grad():
            pred_1 = self.extract_logit_ref(img_1_s)
            pred_1 = F.interpolate(
                pred_1, scale_factor=2, mode="bilinear", align_corners=False
            )
            img_1_warp = F.interpolate(
                img_1_s, scale_factor=0.25, mode="bilinear", align_corners=False
            )

            pred_2 = self.extract_logit_ref(img_2_s)
            pred_2 = F.interpolate(
                pred_2, scale_factor=8, mode="bilinear", align_corners=False
            )
            out_2 = torch.argmax(pred_2, dim=1)

            img_2_s_down = F.interpolate(
                img_2_s, scale_factor=0.25, mode="bilinear", align_corners=False
            )
            img_3_s_down = F.interpolate(
                img_3_s, scale_factor=0.25, mode="bilinear", align_corners=False
            )

        flow_12 = self.flownet(torch.cat([img_2_f, img_1_f], dim=1))
        pred_2_warp = self.warp(pred_1, flow_12)
        img_2_warp = self.warp(img_1_warp, flow_12)

        flow_23 = self.flownet(torch.cat([img_3_f, img_2_f], dim=1))
        pred_3_warp = self.warp(pred_2_warp, flow_23)
        img_3_warp = self.warp(img_2_warp, flow_23)

        with torch.no_grad():
            dm_2 = self.dmnet(img_2_warp, img_2_s_down)
            dm_2 = F.interpolate(
                dm_2, scale_factor=4, mode="bilinear", align_corners=False
            )
            dm_3 = self.dmnet(img_3_warp, img_3_s_down)
            dm_3 = F.interpolate(
                dm_3, scale_factor=4, mode="bilinear", align_corners=False
            )

        loss_semantic = 0.0
        loss_cfnet = 0.0

        # semantic loss
        pred_2_warp = F.interpolate(
            pred_2_warp, scale_factor=4, mode="bilinear", align_corners=False
        )
        loss_semantic += F.cross_entropy(pred_2_warp, out_2, ignore_index=255)

        # cfnet loss
        pred_cc_2 = self.cfnet(img_2_s)
        pred_cc_2 = F.interpolate(
            pred_cc_2, scale_factor=4, mode="bilinear", align_corners=False
        )
        loss = F.cross_entropy(pred_cc_2, out_2, reduction="none", ignore_index=255)
        loss_cfnet += torch.mean(loss * dm_2)

        # semantic loss
        pred_2_final = pred_2_warp * (1 - dm_2) + pred_cc_2 * dm_2
        loss_semantic += F.cross_entropy(pred_2_final, out_2, ignore_index=255)

        # semantic loss
        pred_3_warp = F.interpolate(
            pred_3_warp, scale_factor=4, mode="bilinear", align_corners=False
        )
        loss_semantic += F.cross_entropy(pred_3_warp, gt, ignore_index=255)

        # cfnet loss
        pred_cc_3 = self.cfnet(img_3_s)
        pred_cc_3 = F.interpolate(
            pred_cc_3, scale_factor=4, mode="bilinear", align_corners=False
        )
        loss = F.cross_entropy(pred_cc_3, gt, reduction="none", ignore_index=255)
        loss_cfnet += torch.mean(loss * dm_3)

        # semantic loss
        pred_3_final = pred_3_warp * (1 - dm_3) + pred_cc_3 * dm_3
        loss_semantic += F.cross_entropy(pred_3_final, gt, ignore_index=255)

        loss_semantic /= 4
        loss_cfnet /= 2

        losses = dict()
        loss = dict()
        loss["loss_semantic"] = loss_semantic
        loss["loss_cfnet"] = loss_cfnet
        losses.update(add_prefix(loss, "decode"))

        return losses

    def predict(
        self, inputs: Tensor, inputs_flow: Tensor, data_samples: OptSampleList = None
    ) -> SampleList:
        num_unsup_frames = inputs.shape[2]
        inputs = list(torch.chunk(inputs, num_unsup_frames, dim=2))
        inputs_flow = list(torch.chunk(inputs_flow, num_unsup_frames, dim=2))
        for i in range(num_unsup_frames):
            inputs[i] = inputs[i].squeeze(2)
            inputs_flow[i] = inputs_flow[i].squeeze(2)

        pred = self.extract_logit_ref(inputs[0])
        pred = F.interpolate(pred, scale_factor=2, mode="bilinear", align_corners=False)
        img_warp = F.interpolate(
            inputs[0], scale_factor=0.25, mode="bilinear", align_corners=False
        )
        for i in range(num_unsup_frames - 1):
            flow = self.flownet(torch.cat([inputs_flow[i + 1], inputs_flow[i]], dim=1))
            pred = self.warp(pred, flow)
            img_warp = self.warp(img_warp, flow)
        pred = F.interpolate(pred, scale_factor=4, mode="bilinear", align_corners=True)

        img_cur = F.interpolate(
            inputs[-1], scale_factor=0.25, mode="bilinear", align_corners=False
        )
        dm = self.dmnet(img_warp, img_cur)
        dm = F.interpolate(dm, scale_factor=4, mode="bilinear", align_corners=True)

        pred_cc = self.cfnet(inputs[-1])
        pred_cc = F.interpolate(
            pred_cc, scale_factor=4, mode="bilinear", align_corners=True
        )

        pred_merge = pred * (1 - dm) + pred_cc * dm

        return self.postprocess_result(pred_merge, data_samples)

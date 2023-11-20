# Copyright (c) OpenMMLab. All rights reserved.
import torch
import warnings

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample
from mmseg.datasets.transforms.formatting import PackSegInputs


@TRANSFORMS.register_module()
class PackSegInputs_Accel(PackSegInputs):
    def transfrom_img(self, img):
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        if not img.flags.c_contiguous:
            img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
        else:
            img = img.transpose(2, 0, 1)
            img = to_tensor(img).contiguous()
        return img

    def transfrom_seq(self, results):
        img_list = []
        for i in range(len(results)):
            img = self.transfrom_img(results[i]["img"])
            img = img.unsqueeze(1)
            img_list.append(img)
        img = torch.cat(img_list, dim=1)
        return img

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        packed_results["inputs"] = self.transfrom_seq(results)
        data_sample = SegDataSample()
        if "gt_seg_map" in results[-1]:
            if len(results[-1]["gt_seg_map"].shape) == 2:
                data = to_tensor(results[-1]["gt_seg_map"][None, ...].astype(np.int64))
            else:
                warnings.warn(
                    "Please pay attention your ground truth "
                    "segmentation map, usually the segmentation "
                    "map is 2D, but got "
                    f'{results[-1]["gt_seg_map"].shape}'
                )
                data = to_tensor(results[-1]["gt_seg_map"].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        img_meta = {}
        for key in self.meta_keys:
            if key in results[-1]:
                img_meta[key] = results[-1][key]
        data_sample.set_metainfo(img_meta)
        packed_results["data_samples"] = data_sample

        return packed_results

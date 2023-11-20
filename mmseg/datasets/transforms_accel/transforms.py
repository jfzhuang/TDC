# Copyright (c) OpenMMLab. All rights reserved.
import random
import copy
import numpy as np

import mmcv
from mmseg.registry import TRANSFORMS
from mmseg.datasets.transforms.transforms import RandomCrop, PhotoMetricDistortion
from mmcv.transforms.processing import RandomFlip


@TRANSFORMS.register_module()
class RandomCrop_Accel(RandomCrop):
    def transform(self, results: dict) -> dict:
        crop_bbox = self.crop_bbox(results[-1])
        for i in range(len(results)):
            img = results[i]["img"]
            img = self.crop(img, crop_bbox)
            results[i]["img"] = img
            results[i]["img_shape"] = img.shape[:2]

        results[-1]["gt_seg_map"] = self.crop(results[-1]["gt_seg_map"], crop_bbox)

        return results


@TRANSFORMS.register_module()
class RandomFlip_Accel(RandomFlip):
    def _flip_labeled(self, results: dict, flip_direction) -> None:
        for i in range(len(results)):
            results[i]["img"] = mmcv.imflip(results[i]["img"], direction=flip_direction)

        if results[-1].get("gt_seg_map", None) is not None:
            results[-1]["gt_seg_map"] = self._flip_seg_map(
                results[-1]["gt_seg_map"], direction=flip_direction
            )

    def _flip_on_direction(self, results: dict) -> None:
        cur_dir = self._choose_direction()
        if cur_dir is not None:
            self._flip_labeled(results, cur_dir)

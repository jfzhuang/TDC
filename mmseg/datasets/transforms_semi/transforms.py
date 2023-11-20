# Copyright (c) OpenMMLab. All rights reserved.
import random
import copy
import numpy as np

import mmcv
from mmseg.registry import TRANSFORMS
from mmseg.datasets.transforms.transforms import RandomCrop, PhotoMetricDistortion
from mmcv.transforms.processing import RandomFlip


@TRANSFORMS.register_module()
class RandomCrop_Semi(RandomCrop):
    def generate_crop_bbox(self, img: np.ndarray) -> tuple:
        """Randomly get a crop bounding box.

        Args:
            img (np.ndarray): Original input image.

        Returns:
            tuple: Coordinates of the cropped image.
        """

        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def transform(self, results: dict) -> dict:
        results_labeled, results_unlabeled = results

        img = results_labeled["img"]
        crop_bbox = self.crop_bbox(results_labeled)
        img = self.crop(img, crop_bbox)
        for key in results_labeled.get("seg_fields", []):
            results_labeled[key] = self.crop(results_labeled[key], crop_bbox)
        results_labeled["img"] = img
        results_labeled["img_shape"] = img.shape[:2]

        crop_bbox = self.generate_crop_bbox(results_unlabeled[0]["img"])
        for i in range(len(results_unlabeled)):
            img = results_unlabeled[i]["img"]
            img = self.crop(img, crop_bbox)
            results_unlabeled[i]["img"] = img
            results_unlabeled[i]["img_shape"] = img.shape[:2]

        return results_labeled, results_unlabeled


@TRANSFORMS.register_module()
class RandomFlip_Semi(RandomFlip):
    def _flip_unlabeled(self, results: dict, flip_direction) -> None:
        for i in range(len(results)):
            results[i]["img"] = mmcv.imflip(results[i]["img"], direction=flip_direction)

    def _flip_on_direction(self, results: dict) -> None:
        results_labeled, results_unlabeled = results

        cur_dir = self._choose_direction()
        if cur_dir is None:
            results_labeled["flip"] = False
            results_labeled["flip_direction"] = None
        else:
            results_labeled["flip"] = True
            results_labeled["flip_direction"] = cur_dir
            self._flip(results_labeled)

        cur_dir = self._choose_direction()
        if cur_dir is not None:
            self._flip_unlabeled(results_unlabeled, cur_dir)


@TRANSFORMS.register_module()
class PhotoMetricDistortion_Semi(PhotoMetricDistortion):
    def augment(self, img):
        img = self.brightness(img)
        mode = random.randint(0, 1)
        if mode == 1:
            img = self.contrast(img)
        img = self.saturation(img)
        img = self.hue(img)
        if mode == 0:
            img = self.contrast(img)
        return img

    def transform(self, results: dict) -> dict:
        results_labeled, results_unlabeled = results
        results_unlabeled_s = copy.deepcopy(results_unlabeled)

        for i in range(len(results_unlabeled_s)):
            img = results_unlabeled_s[i]["img"]
            img = self.augment(img)
            results_unlabeled_s[i]["img"] = img
        return results_labeled, results_unlabeled, results_unlabeled_s

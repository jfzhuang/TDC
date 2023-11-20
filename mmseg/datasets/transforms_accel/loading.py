# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional

import mmcv
import mmengine.fileio as fileio
import numpy as np

from mmcv.transforms import LoadImageFromFile

from mmseg.registry import TRANSFORMS
from mmseg.datasets.transforms.loading import LoadAnnotations


@TRANSFORMS.register_module()
class LoadImageFromFile_Accel(LoadImageFromFile):
    def transform(self, results: dict) -> Optional[dict]:
        for i in range(len(results)):
            tmp = results[i]
            filename = tmp["img_path"]
            img_bytes = fileio.get(filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend
            )
            tmp["img"] = img
            tmp["img_shape"] = img.shape[:2]
            tmp["ori_shape"] = img.shape[:2]
            results[i] = tmp

        return results


@TRANSFORMS.register_module()
class LoadAnnotations_Accel(LoadAnnotations):
    def _load_seg_map(self, results: dict) -> None:
        img_bytes = fileio.get(
            results[-1]["seg_map_path"], backend_args=self.backend_args
        )
        gt_semantic_seg = (
            mmcv.imfrombytes(img_bytes, flag="unchanged", backend=self.imdecode_backend)
            .squeeze()
            .astype(np.uint8)
        )

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results[-1]["reduce_zero_label"]
        assert self.reduce_zero_label == results[-1]["reduce_zero_label"], (
            "Initialize dataset with `reduce_zero_label` as "
            f'{results[-1]["reduce_zero_label"]} but when load annotation '
            f"the `reduce_zero_label` is {self.reduce_zero_label}"
        )
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results[-1].get("label_map", None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results[-1]["label_map"].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results[-1]["gt_seg_map"] = gt_semantic_seg
        results[-1]["seg_fields"].append("gt_seg_map")

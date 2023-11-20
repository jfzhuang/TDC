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
class PackSegInputs_Semi(PackSegInputs):
    def transfrom_img(self, img):
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        if not img.flags.c_contiguous:
            img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
        else:
            img = img.transpose(2, 0, 1)
            img = to_tensor(img).contiguous()
        return img

    def transfrom_unlabeled_img(self, results):
        img_list = []
        for i in range(len(results)):
            img = self.transfrom_img(results[i]["img"])
            img = img.unsqueeze(1)
            img_list.append(img)
        img = torch.cat(img_list, dim=1)
        return img

    def transform(self, results: dict) -> dict:
        results_labeled, results_unlabeled, results_unlabeled_s = results

        packed_results = dict()

        packed_results["inputs_labeled"] = self.transfrom_img(results_labeled["img"])
        packed_results["inputs_unlabeled"] = self.transfrom_unlabeled_img(
            results_unlabeled
        )
        packed_results["inputs_unlabeled_s"] = self.transfrom_unlabeled_img(
            results_unlabeled_s
        )
        # print("inputs_labeled:", packed_results["inputs_labeled"].shape)
        # print("inputs_unlabeled:", packed_results["inputs_unlabeled"].shape)
        # print("inputs_unlabeled_s:", packed_results["inputs_unlabeled_s"].shape)

        data_sample = SegDataSample()
        if "gt_seg_map" in results_labeled:
            if len(results_labeled["gt_seg_map"].shape) == 2:
                data = to_tensor(
                    results_labeled["gt_seg_map"][None, ...].astype(np.int64)
                )
            else:
                warnings.warn(
                    "Please pay attention your ground truth "
                    "segmentation map, usually the segmentation "
                    "map is 2D, but got "
                    f'{results_labeled["gt_seg_map"].shape}'
                )
                data = to_tensor(results_labeled["gt_seg_map"].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        img_meta = {}
        for key in self.meta_keys:
            if key in results_labeled:
                img_meta[key] = results_labeled[key]
        data_sample.set_metainfo(img_meta)
        packed_results["data_samples"] = data_sample

        return packed_results

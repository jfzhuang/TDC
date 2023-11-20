# Copyright (c) OpenMMLab. All rights reserved.
import copy
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence

import torch
from mmengine.model import BaseDataPreprocessor

from mmseg.registry import MODELS
from mmseg.utils import stack_batch


@MODELS.register_module()
class SegDataPreProcessor(BaseDataPreprocessor):
    """Image pre-processor for segmentation tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.


    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the input size with defined ``pad_val``, and pad seg map
        with defined ``seg_pad_val``.
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        padding_mode (str): Type of padding. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
        test_cfg (dict, optional): The padding size config in testing, if not
            specify, will use `size` and `size_divisor` params as default.
            Defaults to None, only supports keys `size` or `size_divisor`.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        assert not (
            bgr_to_rgb and rgb_to_bgr
        ), "`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time"
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, (
                "To enable the normalization in "
                "preprocessing, please specify both "
                "`mean` and `std`."
            )
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer("std", torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data["inputs"]
        data_samples = data.get("data_samples", None)

        # TODO: whether normalize should be after stack_batch
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        if training:
            assert data_samples is not None, (
                "During training, ",
                "`data_samples` must be define.",
            )
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val,
            )

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(inputs, data_samples)
        else:
            assert len(inputs) == 1, (
                "Batch inference is not support currently, "
                "as the image size might be different in a batch"
            )
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get("size", None),
                    size_divisor=self.test_cfg.get("size_divisor", None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val,
                )
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

        return dict(inputs=inputs, data_samples=data_samples)


@MODELS.register_module()
class SemiSegDataPreProcessor(SegDataPreProcessor):
    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        data = self.cast_data(data)  # type: ignore

        if training:
            inputs_labeled = data["inputs_labeled"]
            inputs_unlabeled = data["inputs_unlabeled"]
            inputs_unlabeled_s = data["inputs_unlabeled_s"]
            data_samples = data.get("data_samples", None)

            inputs_labeled = [_input[[2, 1, 0], ...] for _input in inputs_labeled]
            inputs_labeled = [_input.float() for _input in inputs_labeled]
            inputs_unlabeled = [_input[[2, 1, 0], ...] for _input in inputs_unlabeled]
            inputs_unlabeled = [_input.float() for _input in inputs_unlabeled]
            inputs_unlabeled_s = [
                _input[[2, 1, 0], ...] for _input in inputs_unlabeled_s
            ]
            inputs_unlabeled_s = [_input.float() for _input in inputs_unlabeled_s]

            if self._enable_normalize:
                inputs_labeled = [
                    (_input - self.mean) / self.std for _input in inputs_labeled
                ]
                inputs_unlabeled = [
                    (_input - self.mean.unsqueeze(-1)) / self.std.unsqueeze(-1)
                    for _input in inputs_unlabeled
                ]
                inputs_unlabeled_s = [
                    (_input - self.mean.unsqueeze(-1)) / self.std.unsqueeze(-1)
                    for _input in inputs_unlabeled_s
                ]

            inputs_labeled = torch.stack(inputs_labeled, dim=0)
            inputs_unlabeled = torch.stack(inputs_unlabeled, dim=0)
            inputs_unlabeled_s = torch.stack(inputs_unlabeled_s, dim=0)

            return dict(
                inputs_labeled=inputs_labeled,
                inputs_unlabeled=inputs_unlabeled,
                inputs_unlabeled_s=inputs_unlabeled_s,
                data_samples=data_samples,
            )

        else:
            inputs = data["inputs"]
            data_samples = data.get("data_samples", None)

            # TODO: whether normalize should be after stack_batch
            if self.channel_conversion and inputs[0].size(0) == 3:
                inputs = [_input[[2, 1, 0], ...] for _input in inputs]

            inputs = [_input.float() for _input in inputs]
            if self._enable_normalize:
                inputs = [(_input - self.mean) / self.std for _input in inputs]

            assert len(inputs) == 1, (
                "Batch inference is not support currently, "
                "as the image size might be different in a batch"
            )
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get("size", None),
                    size_divisor=self.test_cfg.get("size_divisor", None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val,
                )
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

            return dict(inputs=inputs, data_samples=data_samples)


@MODELS.register_module()
class AccelSegDataPreProcessor(SegDataPreProcessor):
    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        data = self.cast_data(data)  # type: ignore

        inputs = data["inputs"]
        data_samples = data.get("data_samples", None)
        inputs_flow = copy.deepcopy(inputs)

        inputs = [_input[[2, 1, 0], ...] for _input in inputs]
        inputs = [_input.float() for _input in inputs]
        inputs = [
            (_input - self.mean.unsqueeze(-1)) / self.std.unsqueeze(-1)
            for _input in inputs
        ]
        inputs = torch.stack(inputs, dim=0)

        inputs_flow = [_input.float() for _input in inputs_flow]
        inputs_flow = [_input / 255.0 for _input in inputs_flow]
        inputs_flow = torch.stack(inputs_flow, dim=0)

        return dict(inputs=inputs, inputs_flow=inputs_flow, data_samples=data_samples)

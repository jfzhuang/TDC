# Copyright (c) OpenMMLab. All rights reserved.
import gc
import copy
import random
import pickle
import numpy as np
import os.path as osp
from typing import List, Tuple

import mmengine
from mmengine.dataset import force_full_init
from mmengine.logging import print_log
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CityscapesDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    METAINFO = dict(
        classes=(
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ),
        palette=[
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
        ],
    )

    def __init__(
        self,
        img_suffix="_leftImg8bit.png",
        seg_map_suffix="_gtFine_labelTrainIds.png",
        **kwargs,
    ) -> None:
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


@DATASETS.register_module()
class CityscapesSemiDataset(CityscapesDataset):
    def __init__(
        self, split_labeled, split_unlabeled, num_unsup_frames, idx_sup=19, **kwargs
    ) -> None:
        self.clip_length = 30
        self.idx_sup = idx_sup
        self.num_unsup_frames = num_unsup_frames
        self.split_labeled = split_labeled
        self.split_unlabeled = split_unlabeled
        if kwargs.get("data_root"):
            self.split_labeled = osp.join(kwargs["data_root"], self.split_labeled)
            self.split_unlabeled = osp.join(kwargs["data_root"], self.split_unlabeled)

        super().__init__(serialize_data=False, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list_labeled, data_list_unlabeled = [], []
        img_dir = self.data_prefix.get("img_path", None)
        ann_dir = self.data_prefix.get("seg_map_path", None)

        lines = mmengine.list_from_file(
            self.split_labeled, backend_args=self.backend_args
        )
        for i in range(len(lines) // self.clip_length):
            video_lines = lines[i * self.clip_length : (i + 1) * self.clip_length]
            data_info_list = []
            for j, line in enumerate(video_lines):
                img_name = line.strip()
                data_info = dict(img_path=osp.join(img_dir, img_name + self.img_suffix))
                if j == self.idx_sup:
                    seg_map = img_name + self.seg_map_suffix
                    data_info["seg_map_path"] = osp.join(ann_dir, seg_map)
                    data_info["label_map"] = self.label_map
                    data_info["reduce_zero_label"] = self.reduce_zero_label
                    data_info["seg_fields"] = []
                data_info_list.append(data_info)
            data_list_labeled.append(data_info_list)
        print_log(f"Loaded {len(data_list_labeled)} labeled clips", logger="current")

        lines = mmengine.list_from_file(
            self.split_unlabeled, backend_args=self.backend_args
        )
        for i in range(len(lines) // self.clip_length):
            video_lines = lines[i * self.clip_length : (i + 1) * self.clip_length]
            data_info_list = []
            for j, line in enumerate(video_lines):
                img_name = line.strip()
                data_info = dict(img_path=osp.join(img_dir, img_name + self.img_suffix))
                data_info_list.append(data_info)
            data_list_unlabeled.append(data_info_list)
        print_log(
            f"Loaded {len(data_list_unlabeled)} unlabeled clips", logger="current"
        )

        return data_list_labeled, data_list_unlabeled

    def full_init(self):
        if self._fully_initialized:
            return
        self.data_list_labeled, self.data_list_unlabeled = self.load_data_list()
        self._fully_initialized = True

    @force_full_init
    def __len__(self) -> int:
        return len(self.data_list_labeled)

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index and automatically call ``full_init`` if the
        dataset has not been fully initialized.

        Args:
            idx (int): The index of data.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        video_idx_labeled = random.choice(list(range(len(self.data_list_labeled))))
        idx_labeled = self.idx_sup
        data_info_labeled = copy.deepcopy(
            self.data_list_labeled[video_idx_labeled][idx_labeled]
        )

        video_idx_unlabeled = random.choice(list(range(len(self.data_list_unlabeled))))
        img_idx_unsup_list = [i for i in range(self.clip_length)]
        random.shuffle(img_idx_unsup_list)
        img_idx_unsup_list = img_idx_unsup_list[: self.num_unsup_frames]
        img_idx_unsup_list = sorted(img_idx_unsup_list)
        data_info_unlabeled = []
        for i in range(self.num_unsup_frames):
            data_info_unlabeled.append(
                copy.deepcopy(
                    self.data_list_unlabeled[video_idx_unlabeled][img_idx_unsup_list[i]]
                )
            )

        return data_info_labeled, data_info_unlabeled


@DATASETS.register_module()
class CityscapesAccelDataset(CityscapesDataset):
    def __init__(self, split_labeled, **kwargs) -> None:
        self.split_labeled = split_labeled
        if kwargs.get("data_root"):
            self.split_labeled = osp.join(kwargs["data_root"], self.split_labeled)

        super().__init__(serialize_data=False, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list_labeled = []
        img_dir = self.data_prefix.get("img_path", None)
        ann_dir = self.data_prefix.get("seg_map_path", None)

        lines = mmengine.list_from_file(
            self.split_labeled, backend_args=self.backend_args
        )
        for line in lines:
            img_name = line.strip()

            data_info_list = []
            frame_id = int(img_name[-6:])
            frame_prefix = img_name[:-7]
            for j in range(5):
                img_name_tmp = "{}_{:06d}".format(frame_prefix, frame_id - 4 + j)
                data_info = dict(
                    img_path=osp.join(img_dir, img_name_tmp + self.img_suffix)
                )
                data_info_list.append(data_info)
            seg_map = img_name_tmp + self.seg_map_suffix
            data_info_list[-1]["seg_map_path"] = osp.join(ann_dir, seg_map)
            data_info_list[-1]["label_map"] = self.label_map
            data_info_list[-1]["reduce_zero_label"] = self.reduce_zero_label
            data_info_list[-1]["seg_fields"] = []
            data_list_labeled.append(data_info_list)
        print_log(f"Loaded {len(data_list_labeled)} labeled clips", logger="current")

        return data_list_labeled

    def full_init(self):
        if self._fully_initialized:
            return
        self.data_list_labeled = self.load_data_list()
        self._fully_initialized = True

    @force_full_init
    def __len__(self) -> int:
        return len(self.data_list_labeled)

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        return self.data_list_labeled[idx]


@DATASETS.register_module()
class CityscapesDFFDataset(CityscapesAccelDataset):
    def __init__(self, validation, **kwargs) -> None:
        self.validation = validation
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list_labeled = []
        img_dir = self.data_prefix.get("img_path", None)
        ann_dir = self.data_prefix.get("seg_map_path", None)

        lines = mmengine.list_from_file(
            self.split_labeled, backend_args=self.backend_args
        )
        for line in lines:
            img_name = line.strip()

            data_info_list = []
            frame_id = int(img_name[-6:])
            frame_prefix = img_name[:-7]

            if self.validation:
                idx_list = [random.randint(0, 8), 9]
            else:
                idx_list = [5, 9]

            for j in idx_list:
                img_name_tmp = "{}_{:06d}".format(frame_prefix, frame_id - 9 + j)
                data_info = dict(
                    img_path=osp.join(img_dir, img_name_tmp + self.img_suffix)
                )
                data_info_list.append(data_info)

            seg_map = img_name_tmp + self.seg_map_suffix
            data_info_list[-1]["seg_map_path"] = osp.join(ann_dir, seg_map)
            data_info_list[-1]["label_map"] = self.label_map
            data_info_list[-1]["reduce_zero_label"] = self.reduce_zero_label
            data_info_list[-1]["seg_fields"] = []
            data_list_labeled.append(data_info_list)
        print_log(f"Loaded {len(data_list_labeled)} labeled clips", logger="current")

        return data_list_labeled


@DATASETS.register_module()
class CityscapesDMNetDataset(CityscapesAccelDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list_labeled = []
        img_dir = self.data_prefix.get("img_path", None)
        ann_dir = self.data_prefix.get("seg_map_path", None)

        lines = mmengine.list_from_file(
            self.split_labeled, backend_args=self.backend_args
        )
        for line in lines:
            img_name = line.strip()

            data_info_list = []
            frame_id = int(img_name[-6:])
            frame_prefix = img_name[:-7]

            idx_list = [i for i in range(9)]
            random.shuffle(idx_list)
            idx_list = idx_list[:2]
            idx_list.append(9)
            idx_list = sorted(idx_list)

            for j in idx_list:
                img_name_tmp = "{}_{:06d}".format(frame_prefix, frame_id - 9 + j)
                data_info = dict(
                    img_path=osp.join(img_dir, img_name_tmp + self.img_suffix)
                )
                data_info_list.append(data_info)

            seg_map = img_name_tmp + self.seg_map_suffix
            data_info_list[-1]["seg_map_path"] = osp.join(ann_dir, seg_map)
            data_info_list[-1]["label_map"] = self.label_map
            data_info_list[-1]["reduce_zero_label"] = self.reduce_zero_label
            data_info_list[-1]["seg_fields"] = []
            data_list_labeled.append(data_info_list)
        print_log(f"Loaded {len(data_list_labeled)} labeled clips", logger="current")

        return data_list_labeled


@DATASETS.register_module()
class CityscapesSCNetDataset(CityscapesAccelDataset):
    def __init__(self, validation, **kwargs) -> None:
        self.validation = validation
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list_labeled = []
        img_dir = self.data_prefix.get("img_path", None)
        ann_dir = self.data_prefix.get("seg_map_path", None)

        lines = mmengine.list_from_file(
            self.split_labeled, backend_args=self.backend_args
        )
        for line in lines:
            img_name = line.strip()

            data_info_list = []
            frame_id = int(img_name[-6:])
            frame_prefix = img_name[:-7]

            if self.validation:
                idx_list = [5, 6, 7, 8, 9]
            else:
                idx_list = [i for i in range(9)]
                random.shuffle(idx_list)
                idx_list = idx_list[:2]
                idx_list.append(9)
                idx_list = sorted(idx_list)

            for j in idx_list:
                img_name_tmp = "{}_{:06d}".format(frame_prefix, frame_id - 9 + j)
                data_info = dict(
                    img_path=osp.join(img_dir, img_name_tmp + self.img_suffix)
                )
                data_info_list.append(data_info)

            seg_map = img_name_tmp + self.seg_map_suffix
            data_info_list[-1]["seg_map_path"] = osp.join(ann_dir, seg_map)
            data_info_list[-1]["label_map"] = self.label_map
            data_info_list[-1]["reduce_zero_label"] = self.reduce_zero_label
            data_info_list[-1]["seg_fields"] = []
            data_list_labeled.append(data_info_list)
        print_log(f"Loaded {len(data_list_labeled)} labeled clips", logger="current")

        return data_list_labeled

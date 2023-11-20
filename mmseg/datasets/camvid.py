from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
from mmseg.datasets.cityscapes import (
    CityscapesDataset,
    CityscapesSemiDataset,
    CityscapesAccelDataset,
)


@DATASETS.register_module()
class CamVidDataset(CityscapesDataset):
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
        ],
    )

    def __init__(self, img_suffix=".png", seg_map_suffix=".png", **kwargs) -> None:
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


@DATASETS.register_module()
class CamVidSemiDataset(CityscapesSemiDataset):
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
        ],
    )

    def __init__(
        self, img_suffix=".png", seg_map_suffix=".png", idx_sup=29, **kwargs
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            idx_sup=idx_sup,
            **kwargs
        )


@DATASETS.register_module()
class CamVidAccelDataset(CityscapesAccelDataset):
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
        ],
    )

    def __init__(self, img_suffix=".png", seg_map_suffix=".png", **kwargs) -> None:
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

# dataset settings
data_root = "data/cityscapes/"
crop_size = (256, 512)
train_pipeline = [
    dict(type="LoadImageFromFile_Semi"),
    dict(type="LoadAnnotations_Semi"),
    dict(type="RandomCrop_Semi", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip_Semi", prob=0.5),
    dict(type="PhotoMetricDistortion_Semi"),
    dict(type="PackSegInputs_Semi"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 512), keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="CityscapesSemiDataset",
        data_root=data_root,
        data_prefix=dict(
            img_path="leftImg8bit_sequence_down_2x/train",
            seg_map_path="gtFine_down_2x/train",
        ),
        split_labeled="splits/train_unsup_1-30.txt",
        split_unlabeled="splits/train_unsup_all.txt",
        num_unsup_frames=5,
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CityscapesDataset",
        data_root=data_root,
        data_prefix=dict(
            img_path="leftImg8bit_sequence_down_2x/val",
            seg_map_path="gtFine_down_2x/val",
        ),
        ann_file="splits/val.txt",
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator

# optimizer
optimizer = dict(
    type="SGD",
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001,
)
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={"decode_head": dict(lr_mult=10.0)}),
    clip_grad=None,
)
# learning policy
param_scheduler = [
    dict(type="PolyLR", eta_min=1e-4, power=0.9, begin=0, end=40000, by_epoch=False)
]
# runtime settings
train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=1000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, save_best="mIoU"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)

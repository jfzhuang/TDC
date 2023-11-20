_base_ = [
    "../../../configs/_base_/models/pspnet_r18-d8_v2.py",
    "../../../configs/_base_/datasets/cityscapes_semi.py",
    "../../../configs/_base_/default_runtime.py",
    "../../../configs/_base_/schedules/schedule.py",
]

train_dataloader = dict(dataset=dict(num_unsup_frames=5))
data_preprocessor = dict(type="SemiSegDataPreProcessor")

optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={"prototype_global": dict(lr_mult=10.0)})
)

model = dict(type="Tem_Proto_v1", weight_unsup=0.2)
seed = 666
deterministic = True

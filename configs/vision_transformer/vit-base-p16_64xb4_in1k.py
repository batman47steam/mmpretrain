_base_ = [
    '../_base_/models/vit-base-p16.py',
    '../_base_/datasets/imagenet_bs64_pil_resize_autoaug.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py'
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
)

# model setting
model = dict(
    head=dict(hidden_dim=3072),
    train_cfg=dict(augments=dict(type='Mixup', alpha=0.2)),
)

# schedule setting
optim_wrapper = dict(clip_grad=dict(max_norm=1.0))

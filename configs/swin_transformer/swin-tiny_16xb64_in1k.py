_base_ = [
    '../_base_/models/swin_transformer/tiny_224.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(pad_small_map=True, img_size=128)
)
# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))

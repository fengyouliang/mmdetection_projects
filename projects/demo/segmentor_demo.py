import os

available_gpu_ids = [3]
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(list(map(str, available_gpu_ids)))

from mmseg.models.builder import build_segmentor
from mmcv import Config


class config:
    # model settings
    norm_cfg = dict(type='BN', requires_grad=True)
    model = dict(
        type='EncoderDecoder',
        pretrained=None,
        backbone=dict(
            type='ResNeSt',
            stem_channels=128,
            radix=2,
            reduction_factor=4,
            avg_down_stride=True,
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True),
        decode_head=dict(
            type='PSPHead',
            in_channels=2048,
            in_index=3,
            channels=512,
            pool_scales=(1, 2, 3, 6),
            dropout_ratio=0.1,
            num_classes=8,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                class_weight=[0.85623873, 0.49506696, 1.47724937, 1.34005817, 0.68560289, 1.24642737, 0.41255728,
                              1.48679921])),
        auxiliary_head=dict(
            type='FCNHead',
            in_channels=1024,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=8,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))
    # model training and testing settings
    train_cfg = dict()
    test_cfg = dict(mode='whole')


if __name__ == '__main__':
    segmentor = build_segmentor(config.model, Config(config.train_cfg), Config(config.test_cfg))
    print(segmentor)

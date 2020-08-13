import os
available_gpu_ids = [1]
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(list(map(str, available_gpu_ids)))

from mmdet.models.builder import build_backbone, build_detector, build_neck
from mmcv import Config
import module.models.backbone.senet
import torch


class config:
    neck = dict(
        type='FPN',
        in_channels=[8, 16, 32, 64],
        out_channels=8,
        num_outs=5,
        norm_cfg=dict(type='BN', requires_grad=True),  # FPN with BN
        upsample_cfg=dict(
            type='carafe',
            up_kernel=5,
            up_group=1,
            encoder_kernel=3,
            encoder_dilation=1,
            compressed_channels=64
        ),

    )


if __name__ == '__main__':
    # cfg = Config(config.neck)
    s = config.neck['in_channels'][-1]
    in_channels = config.neck['in_channels']
    feat_sizes = [s // 2**i for i in range(4)]  # [64, 32, 16, 8]
    out_channels = config.neck['out_channels']

    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels))
    ]
    module = build_neck(config.neck)
    module = module.cuda()
    feats = [item.cuda() for item in feats]
    output = module(feats)
    output_size = [item.size() for item in output]
    print(module)

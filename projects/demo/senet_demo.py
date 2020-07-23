from mmdet.models.builder import build_backbone, build_detector
from mmcv import Config
import module.models.backbone.senet


class config:
    # model settings
    model = dict(
        type='FasterRCNN',
        pretrained=None,
        backbone=dict(
            type='SENet',
            block='SEResNetBottleneck',
            layers=[3, 4, 6, 3],
            groups=1,
            reduction=16,
            dropout_p=None,
            inplanes=64,
            input_3x3=False,
            downsample_kernel_size=1,
            downsample_padding=0,
            num_classes=1
        ),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0))))
    # model training and testing settings
    train_cfg = dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False))
    test_cfg = dict(
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))


if __name__ == '__main__':
    detector = build_detector(config.model, Config(config.train_cfg), Config(config.test_cfg))
    print(detector)

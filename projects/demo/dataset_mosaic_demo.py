import module
from mmdet.datasets import build_dataset, build_dataloader

classes = ('knife', 'scissors', 'lighter', 'zippooil', 'pressure', 'slingshot', 'handcuffs', 'nailpolish', 'powerbank', 'firecrackers')
dataset_type = 'Mosaic_CocoDataset'
# dataset_type = 'CocoDataset'
data_root = '/fengyouliang/datasets/x-ray/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),

    dict(type='LoadMosaicImageAndAnnotations', image_shape=[512, 800]),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/fold0/train.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline)
)

dataset = [build_dataset(data['train'])]

print(dataset)
bs = 4
data_loaders = [
    build_dataloader(
        ds,
        bs,
        0,
        # cfg.gpus will be ignored if distributed
        len(range(0, 1)),
        dist=False,
        seed=41) for ds in dataset
]
data_loader = data_loaders[0]

a = iter(data_loader).__next__()

import numpy as np
import matplotlib.pyplot as plt


images = a['img'].data[0].numpy()
bboxes = a['gt_bboxes'].data
labels = a['gt_labels'].data

for image, bbox, label in zip(images, bboxes, labels):
    image = image.transpose(1, 2, 0)
    for _box, _label in zip(bbox, label):
        print()
    print()


# plt.imshow(image)
# for box in bboxes[0]:
#     x1, y1, x2, y2 = box
#     plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='r', linewidth=3))
# plt.show()

print(data_loader)

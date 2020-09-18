import module
from mmdet.datasets import build_dataset, build_dataloader
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

classes = ('knife', 'scissors', 'lighter', 'zippooil', 'pressure', 'slingshot', 'handcuffs', 'nailpolish', 'powerbank', 'firecrackers')
dataset_type = 'Mosaic_CocoDataset'
# dataset_type = 'CocoDataset'
data_root = '/fengyouliang/datasets/x-ray/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),

    dict(type='LoadMosaicImageAndAnnotations', image_shape=512, not_m_size=(1333, 800)),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='Normalize', **img_norm_cfg),
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
        pipeline=train_pipeline,
        mosaic_ratio=0.5,
    )
)

dataset = [build_dataset(data['train'])]

bs = 8
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

data_per_batch = iter(data_loader).__next__()


images = data_per_batch['img'].data[0].numpy()
bboxes = data_per_batch['gt_bboxes'].data[0]
labels = data_per_batch['gt_labels'].data[0]

for image, bbox, label in zip(images, bboxes, labels):
    image = image.transpose(1, 2, 0)
    plt.imshow(image)
    plt.text(0, 1, f"{image.shape[0]} - {image.shape[1]}")

    for _box, _label in zip(bbox, label):
        x1, y1, x2, y2 = _box
        label_name = classes[_label.data]
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='r', linewidth=1))
        plt.text((x1 + x2) / 2, (y1 + y2) / 2, str(label_name))

    plt.show()

print(data_loader)

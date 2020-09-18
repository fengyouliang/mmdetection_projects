import mmseg_module
from mmseg.datasets import build_dataset, build_dataloader
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# dataset settings
dataset_type = 'PCLMosaicDataset'
data_root = '/fengyouliang/datasets/PCL/'

img_norm_cfg = dict(
    mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
image_size = (256, 256)
train_pipeline = [
    dict(type='LoadMosaicImageAndAnnotations', mosaic_size=image_size, center=0.5, center_range=0.2),
    dict(type='Resize', img_scale=image_size, multiscale_mode="value"),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=image_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/image',
        ann_dir='train/label_cvt',
        split='train/split/val_mini.txt',
        mosaic_ratio=1,
        pipeline=train_pipeline),
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

img_metas = data_per_batch['img_metas']
img_metas_data = img_metas.data[0]

images = data_per_batch['img'].data[0].numpy()
labels = data_per_batch['gt_semantic_seg'].data[0]

for image, label, data_info in zip(images, labels, img_metas_data):
    image = image.transpose(1, 2, 0)
    label = label.squeeze()
    plt.imshow(image / 255)
    plt.show()
    plt.imshow(label)
    plt.show()

print(data_loader)

# convert origin image to mask

import glob
import os
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class opt:
    colors = [
        [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [0, 0, 0],

        [64, 0, 0],
        [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
        [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
        [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
    ]

    matches = [100, 200, 300, 400, 500, 600, 700, 800]


def cvt(label_file, save_path):
    label = cv.imread(label_file, -1)
    # h, w = label.shape
    # label_cvt = np.zeros((*label.shape[:2], 3), dtype=np.uint8)
    label_cvt = np.zeros(label.shape, dtype=np.uint8)

    for idx, match in enumerate(opt.matches):
        # label_cvt[label == match] = tuple(opt.colors[idx])
        label_cvt[label == match] = idx
    assert cv.imwrite(f"{save_path}/{Path(label_file).name}", label_cvt)


def run_cvt():
    label_path = '/fengyouliang/datasets/PCL/train/label'
    all_label_files = glob.glob(f"{label_path}/*.png")
    pbar = tqdm(all_label_files, total=len(all_label_files))

    save_path = '/fengyouliang/datasets/PCL/train/label_cvt_dim1'
    os.makedirs(save_path, exist_ok=True)

    for label_file in pbar:
        pbar.set_description(label_file.split('/')[-1])
        cvt(label_file, save_path)


def check_cvt():
    label_path = '/fengyouliang/datasets/PCL/train/label_cvt'
    all_label_files = glob.glob(f"{label_path}/*.png")
    pbar = tqdm(all_label_files, total=len(all_label_files))

    for label_file in pbar:
        pbar.set_description(label_file)
        # label_cvt = cv.imread(label_file, -1)
        label_cvt = Image.open(label_file)
        print(np.unique(label_cvt))
        print()


def gen_split():
    image_path = '/fengyouliang/datasets/PCL/train/image'
    all_images = glob.glob(f"{image_path}/*.tif")
    train_set, val_set = train_test_split(all_images, test_size=0.05)

    train_name = [Path(item).stem for item in train_set]
    val_name = [Path(item).stem for item in val_set]

    train_name = sorted(train_name, key=len)
    val_name = sorted(val_name, key=len)
    # train_name.sort()
    # val_name.sort()

    train_txt = '/fengyouliang/datasets/PCL/train/split/train.txt'
    val_txt = '/fengyouliang/datasets/PCL/train/split/val.txt'

    with open(train_txt,  'w', encoding='utf-8') as f:
        f.write('\n'.join(train_name))

    with open(val_txt,  'w', encoding='utf-8') as f:
        f.write('\n'.join(val_name))


if __name__ == '__main__':
    run_cvt()
    # gen_split()

import cv2 as cv
from pathlib import Path
import glob
from tqdm import tqdm
from collections import Counter
import numpy as np
from projects.utils import AverageMeter


class opt:
    train_label_root = '/fengyouliang/datasets/PCL/train/label'


def processing_one_image(image_path):
    image = cv.imread(image_path, -1)
    pixel_counter = Counter(image.flatten())
    pixel_percent = {k: v / 256 / 256 for k, v in pixel_counter.items()}
    pixel_percent_all = np.array([pixel_percent.get(k, 0) for k in range(100, 801, 100)])
    assert pixel_percent_all.sum() == 1
    return pixel_percent_all


def class_statistics():
    all_train_images = glob.glob(f"{opt.train_label_root}/*.png")

    pixel_meter = AverageMeter(size=(8,))

    pbar = tqdm(all_train_images, total=len(all_train_images))
    for train_image_path in pbar:
        per_image_result = processing_one_image(train_image_path)
        pixel_meter.update(per_image_result)
        # pbar.set_description(str(pixel_meter))
    print(pixel_meter.avg)


if __name__ == '__main__':
    class_statistics()

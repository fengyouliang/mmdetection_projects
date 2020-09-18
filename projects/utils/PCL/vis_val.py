import glob
import time
import os
from pathlib import Path
import cv2 as cv
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

available_gpu_ids = [0]
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(list(map(str, available_gpu_ids)))

from mmseg.apis import inference_segmentor, init_segmentor
import mmcv


class opt:
    image_path = '/fengyouliang/datasets/PCL/train/image'
    label_path = '/fengyouliang/datasets/PCL/train/label_cvt'
    split_file = '/fengyouliang/datasets/PCL/train/split/val.txt'
    matches = [0, 1, 2, 3, 4, 5, 6, 7]
    PALETTE = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
               [0, 0, 0]]
    plt_color = [[0, 255, 255], ]


def get_val_idxs():
    with open(opt.split_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    val_ids = [item.strip() for item in lines]
    return val_ids


def cvt_id2color(image):
    cvt_image = np.zeros([*image.shape[:2], 3], dtype=np.int)
    for idx, match in enumerate(opt.matches):
        cvt_image[image == match] = opt.PALETTE[idx]

    return cvt_image


def val_one_image(model, image_id):
    ori_image = f"{opt.image_path}/{image_id}.tif"
    ori_gt = f"{opt.label_path}/{image_id}.png"

    img = mmcv.imread(ori_image)
    gt_img = mmcv.imread(ori_gt)
    result = inference_segmentor(model, img)
    result = result[0]

    pred = cvt_id2color(result)
    gt = cvt_id2color(gt_img[..., 0])

    plt.imshow(img[:, :, ::-1])
    plt.show()
    plt.imshow(gt[:, :, ::-1])
    plt.show()
    plt.imshow(pred[:, :, ::-1])
    plt.show()


def get_model(model_path, epoch_name):
    project_type = model_path.split('/')[-2]
    model_type = model_path.split('/')[-1]

    config_file = f'../../configs/{project_type}/{model_type}.py'
    checkpoint_file = f'/fengyouliang/model_output/mmseg_work_dirs/{project_type}/{model_type}/{epoch_name}.pth'

    print(f"project name: {project_type}")
    print(f"model type: {model_type}")
    print(f"loading config form {config_file}")
    print(f"loading checkpoint form {checkpoint_file}")
    print('init model right now')
    start = time.time()
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    end = time.time()
    print(f'loading done! time: {end - start: .4f}s')
    return model


def run_vis(model, id=None, max_num=1):
    val_ids = get_val_idxs()

    for i in range(max_num):

        if id is not None:
            assert isinstance(id, int), f"id type should be int not {type(id)}"
            assert id in val_ids, f"{id} not in val split"
            vis_idx = id
        else:
            vis_idx = np.random.choice(val_ids, 1)[0]

        val_one_image(model, vis_idx)

        if id is not None:
            break


def main():
    model_path = '/fengyouliang/model_output/mmseg_work_dirs/PCL/ocrnet_dist'
    model = get_model(model_path, epoch_name='latest')
    run_vis(model)


if __name__ == '__main__':
    main()

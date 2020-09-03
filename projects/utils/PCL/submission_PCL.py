import glob
import os
from pathlib import Path
import cv2 as cv
import numpy as np
from tqdm import tqdm

available_gpu_ids = [0]
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(list(map(str, available_gpu_ids)))

from mmseg.apis import inference_segmentor, init_segmentor
import mmcv


class opt:
    test_path = '/fengyouliang/datasets/PCL/test'
    matches = [0, 1, 2, 3, 4, 5, 6, 7]


def inference_one_image(model, image, project_type, model_type):
    img = mmcv.imread(image)
    result = inference_segmentor(model, img)
    result = result[0]

    pred = np.zeros(result.shape, dtype=np.uint16)

    for idx, match in enumerate(opt.matches):
        # label_cvt[label == match] = tuple(opt.colors[idx])
        pred[result == match] = (idx + 1) * 100

    save_path = f'../../submission/{project_type}/{model_type}/results'
    os.makedirs(save_path, exist_ok=True)

    out_file = f"{save_path}/{Path(image).stem}.png"
    cv.imwrite(out_file, pred)


def run_submission(model_path, epoch_name):
    project_type = model_path.split('/')[-2]
    model_type = model_path.split('/')[-1]

    config_file = f'../../configs/{project_type}/{model_type}.py'
    checkpoint_file = f'/fengyouliang/model_output/mmseg_work_dirs/{project_type}/{model_type}/{epoch_name}.pth'

    print(f"project name: {project_type}")
    print(f"model type: {model_type}")
    print(f"loading config form {config_file}")
    print(f"loading checkpoint form {checkpoint_file}")
    print('init model right now')
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    print('start inference')

    all_test_images = glob.glob(f"{opt.test_path}/*.tif")
    pbar = tqdm(all_test_images, total=len(all_test_images))

    for image_file in pbar:
        pbar.set_description(Path(image_file).name)
        inference_one_image(model, image_file, project_type, model_type)


def check_submission():
    sub_path = '/workspace/projects/submission/PCL/fcn/'
    test_preds = glob.glob(f"{sub_path}/*.png")

    for test_pred in test_preds:
        pred = cv.imread(test_pred, -1)
        assert set(np.unique(pred)).issubset(set([(i + 1) * 100 for i in range(8)]))
        assert pred.dtype == np.uint16
        print()


def main():
    model_path = '/fengyouliang/model_output/mmseg_work_dirs/PCL/ocrnet_dist'
    run_submission(model_path, epoch_name='latest')


if __name__ == '__main__':
    main()
    # cd submission/project/model_type
    # zip -r results.zip results/

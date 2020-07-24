import json
import os
from tqdm import tqdm

available_gpu_ids = [0]
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(list(map(str, available_gpu_ids)))

from mmdet.apis import init_detector, inference_detector


def main():
    model_path = '/fengyouliang/model_output/work_dirs_multi/x_ray/base_faster_mdconv_c3-c5'
    model_type = model_path.split('/')[-1]
    is_multi = 'multi' in model_path

    run_submission(model_type, is_multi)


def submission_test(config_file, checkpoint_file, test_path, save_path, save_name):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    results = []

    bar = tqdm(sorted(os.listdir(test_path)))
    for file in bar:
        bar.set_description(file)
        img = f'{test_path}/{file}'
        result = inference_detector(model, img)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None

        result = bbox_result
        pre_image_res = [item.tolist() for item in result]
        results.append(pre_image_res)

    dump_ = f'{save_path}/{save_name}.json'
    json.dump(results, open(dump_, 'w'), ensure_ascii=False, indent=4)


def run_submission(model_type, is_multi):
    project = 'x_ray'
    # model_type = 'cascade_r101_64_4d_345'  # config file basename

    config_file = f'../../configs/x_ray/{model_type}.py'

    if is_multi:
        checkpoint_file = f'/fengyouliang/model_output/work_dirs_multi/{project}/{model_type}/latest.pth'
    else:
        checkpoint_file = f'/fengyouliang/model_output/work_dirs/{project}/{model_type}/latest.pth'

    test_path = '/fengyouliang/datasets/x-ray/test1'

    save_path = f'/workspace/projects/submission/{project}'
    save_name = f'{model_type}'

    vis_save_path = f'../../vis_show/{project}/{model_type}'

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(vis_save_path, exist_ok=True)

    submission_test(config_file, checkpoint_file, test_path, save_path, save_name)


if __name__ == '__main__':
    main()

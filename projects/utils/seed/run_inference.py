import json
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np


available_gpu_ids = [0]
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(list(map(str, available_gpu_ids)))


def main():
    model_path = '/fengyouliang/model_output/mmdet_work_dirs/seed/base_cascade_r101_dcn'
    run_submission(model_path, epoch_name='epoch_10')


def run_submission(model_path, epoch_name='latest'):
    project_type = model_path.split('/')[-2]
    model_type = model_path.split('/')[-1]

    config_file = f'../../configs/{project_type}/{model_type}.py'
    checkpoint_file = f'/fengyouliang/model_output/work_dirs/{project_type}/{model_type}/{epoch_name}.pth'

    assert os.path.isfile(config_file), config_file
    assert os.path.isfile(checkpoint_file), checkpoint_file

    test_path = '/fengyouliang/datasets/seed/test'

    save_path = f'/workspace/projects/submission/{project_type}/{model_type}/result'

    vis_save_path = f'../../vis_show/{project_type}/{model_type}'

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(vis_save_path, exist_ok=True)

    print(f"run in config: {config_file}")
    print(f"run in checkpoint: {checkpoint_file}")
    submission_test(config_file, checkpoint_file, test_path, save_path)


def processing_result(result):
    classes = ('window_shielding', 'non_traffic_sign', 'multi_signs')
    ret = []
    for class_id, class_res in enumerate(result):
        category = classes[class_id]
        for bbox_item in class_res:
            x1, y1, x2, y2, score = bbox_item
            ret.append({
                "category": category,
                "x": (x1 + x2) / 2,
                "y": (y1 + y2) / 2,
                "w": x2 - x1,
                "h": y2 - y1,
                "score": score
            })
    return ret


def submission_test(config_file, checkpoint_file, test_path, save_path):
    from mmdet.apis import init_detector, inference_detector

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

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

        json_res = processing_result(bbox_result)

        dump_ = f'{save_path}/{Path(file).with_suffix(".json")}'
        json.dump(json_res, open(dump_, 'w'), ensure_ascii=False, indent=4, cls=NpEncoder)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


if __name__ == '__main__':
    main()

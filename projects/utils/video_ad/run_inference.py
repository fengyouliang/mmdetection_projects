import json
import os
from tqdm import tqdm

available_gpu_ids = [1]
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(list(map(str, available_gpu_ids)))

from mmdet.apis import init_detector, inference_detector


def main():
    model_path = '/fengyouliang/model_output/work_dirs_multi/video_ad/cascade_r101_64_4d'
    model_type = model_path.split('/')[-1]
    is_multi = 'multi' in model_path

    run_submission(model_type, is_multi)


def get_image_sort():
    with open('/fengyouliang/datasets/video_ad/annotations/test_image_sort.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    image_list = [item['file_name'] for item in data]
    return image_list


def submission_test(config_file, checkpoint_file, test_path, save_path, save_name, vis_save_path, save_vis=False):
    image_list = get_image_sort()

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    results = []
    bar = tqdm(image_list)
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

        if save_vis:
            model.show_result(img, result, text_color='red', thickness=3, out_file=f"{vis_save_path}/{file}")

        pre_image_res = [item.tolist() for item in result]
        results.append(pre_image_res)

    dump_ = f'{save_path}/{save_name}.json'
    json.dump(results, open(dump_, 'w'), ensure_ascii=False, indent=4)
    print(dump_)


def run_submission(model_type, is_multi):
    project = 'video_ad'
    # model_type = 'base_faster_r101x'  # config file basename

    config_file = f'../../configs/{project}/{model_type}.py'

    if is_multi:
        checkpoint_file = f'/fengyouliang/model_output/work_dirs_multi/{project}/{model_type}/latest.pth'
    else:
        checkpoint_file = f'/fengyouliang/model_output/work_dirs/{project}/{model_type}/latest.pth'

    test_path = f'/fengyouliang/datasets/video_ad/test_images'

    save_path = f'/workspace/projects/submission/{project}'
    save_name = f'{model_type}'

    vis_save_path = f'../../vis_show/{project}/{model_type}'

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(vis_save_path, exist_ok=True)

    submission_test(config_file, checkpoint_file, test_path, save_path, save_name, vis_save_path, save_vis=False)


if __name__ == '__main__':
    main()

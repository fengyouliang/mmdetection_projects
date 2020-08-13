import json
import os
from tqdm import tqdm

available_gpu_ids = [0]
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(list(map(str, available_gpu_ids)))

from mmdet.apis import init_detector, inference_detector
import mmcv


def main():
    model_path = '/fengyouliang/model_output/work_dirs/x_ray/base_faster_mosaic'
    model_type = model_path.split('/')[-1]
    is_multi = 'multi' in model_path

    run_submission(model_type, is_multi)


def get_categories():
    classes = ('knife', 'scissors', 'lighter', 'zippooil', 'pressure', 'slingshot', 'handcuffs', 'nailpolish', 'powerbank', 'firecrackers')
    categories = [{'id': idx + 1, 'name': name} for idx, name in enumerate(classes)]
    return categories


def pseudo_inference(config_file, checkpoint_file, test_path, save_path, save_name):

    coco = dict()
    coco["info"] = "x-ary detection"
    coco["license"]: ["fengyun"]
    coco['categories'] = get_categories()

    images = []
    annotations = []

    image_index = 0
    annotation_index = 0

    # model = init_detector(config_file, checkpoint_file, device='cuda:0')
    model = init_detector(config_file, checkpoint_file, device='cpu')

    bar = tqdm(sorted(os.listdir(test_path)))
    for file in bar:

        cur_image_index = image_index
        image_index += 1

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

        # get coco info
        image_mmcv = mmcv.imread(img)
        h, w, c = image_mmcv.shape
        image_item = {
            "height": h,
            "width": w,
            "id": cur_image_index,
            "file_name": file
        }
        images.append(image_item)

        for class_idx, bboxes in enumerate(pre_image_res):

            cur_box_index = annotation_index
            annotation_index += 1

            if len(bboxes) == 0:
                continue

            for box in bboxes:
                x1, y1, x2, y2, score = box
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                if score < 0.5:
                    continue

                area = w * h
                ann_item = {
                  "image_id": cur_image_index,
                  "id": cur_box_index,
                  "category_id": class_idx + 1,
                  "bbox": [x, y, w, h],
                  "segmentation": [[x, y, x, y + h, x + w, y + h, x + w, y]],
                  "area": area,
                  "iscrowd": 0,
                }
                annotations.append(ann_item)

    coco['images'] = images
    coco['annotations'] = annotations

    dump_ = f'{save_path}/{save_name}.json'
    json.dump(coco, open(dump_, 'w'), ensure_ascii=False, indent=4)
    print(dump_)


def run_submission(model_type, is_multi):
    project = 'x_ray'

    config_file = f'../configs/{project}/{model_type}.py'

    if is_multi:
        checkpoint_file = f'/fengyouliang/model_output/work_dirs_multi/{project}/{model_type}/latest.pth'
    else:
        checkpoint_file = f'/fengyouliang/model_output/work_dirs/{project}/{model_type}/latest.pth'

    test_path = f'/fengyouliang/datasets/x-ray/test1'

    save_path = f'/fengyouliang/datasets/x-ray/coco/annotations/fold0'
    save_name = f'pseudo_label_{model_type}'

    os.makedirs(save_path, exist_ok=True)

    pseudo_inference(config_file, checkpoint_file, test_path, save_path, save_name)


if __name__ == '__main__':
    main()

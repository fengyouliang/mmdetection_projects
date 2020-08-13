import json


def merge_coco(coco_a, coco_b, save_path, save_name='pseudo_labeling'):
    with open(coco_a, 'r', encoding='utf-8') as fid:
        cocoa = json.load(fid)
    with open(coco_b, 'r', encoding='utf-8') as fid:
        cocob = json.load(fid)

    # assert cocoa['info'] == cocob['info']
    # assert cocoa['license'] == cocob['license']
    assert cocoa['categories'] == cocob['categories']

    images_a = cocoa['images']
    images_b = cocob['images']
    anns_a = cocoa['annotations']
    anns_b = cocob['annotations']

    id_a = [item['id'] for item in images_a]
    id_b = [item['id'] for item in images_b]

    image_id2imagea = {item['id']: item for item in images_a}
    image_id2imageb = {item['id']: item for item in images_b}

    if len(set(id_a) & set(id_b)) != 0:
        pass

    image_id2ann_a = {item: [] for item in id_a}
    image_id2ann_b = {item: [] for item in id_b}

    for item in anns_a:
        image_id2ann_a[item['image_id']].append(item)

    for item in anns_b:
        image_id2ann_b[item['image_id']].append(item)

    merge_coco_json = dict()
    merge_coco_json['info'] = cocoa['info']
    merge_coco_json['license'] = cocoa['license']
    merge_coco_json['categories'] = cocoa['categories']

    merge_images = []
    merge_annotations = []

    merge_image_index = 0
    merge_ann_index = 0

    for image_id, image in image_id2imagea.items():
        ori_image_id = image['id']
        anns = image_id2ann_a[ori_image_id]
        for ann in anns:
            ann['image_id'] = merge_image_index
            ann['id'] = merge_ann_index

            merge_annotations.append(ann)

        image['id'] = merge_image_index
        new_merge_image_item = image
        merge_images.append(new_merge_image_item)

        merge_image_index += 1
        merge_ann_index += 1

    for image_id, image in image_id2imageb.items():
        ori_image_id = image['id']
        anns = image_id2ann_b[ori_image_id]
        for ann in anns:
            ann['image_id'] = merge_image_index
            ann['id'] = merge_ann_index

            merge_annotations.append(ann)

        image['id'] = merge_image_index
        new_merge_image_item = image
        merge_images.append(new_merge_image_item)

        merge_image_index += 1
        merge_ann_index += 1

    assert len(merge_images) == len(images_a) + len(images_b)
    assert len(merge_annotations) == len(anns_a) + len(anns_b)

    merge_coco_json['images'] = merge_images
    merge_coco_json['annotations'] = merge_annotations

    json.dump(merge_coco_json, open(f"{save_path}/{save_name}.json", 'w'), ensure_ascii=False, indent=4)
    print(f"{save_path}/{save_name}.json")


def test_merge():
    from pycocotools.coco import COCO
    train = COCO('/fengyouliang/datasets/x-ray/coco/annotations/fold4/train.json')
    val = COCO('/fengyouliang/datasets/x-ray/coco/annotations/fold4/val.json')
    pseudo = COCO('/fengyouliang/datasets/x-ray/coco/annotations/fold4/pseudo_demo.json')
    print()


def main():
    # a = '/fengyouliang/datasets/x-ray/coco/annotations/fold4/train.json'
    # b = '/fengyouliang/datasets/x-ray/coco/annotations/fold4/val.json'
    # merge_coco(a, b, '/fengyouliang/datasets/x-ray/coco/annotations/fold4', 'pseudo_demo')
    test_merge()


if __name__ == '__main__':
    main()

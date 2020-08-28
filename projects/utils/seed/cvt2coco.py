from pycocotools.coco import COCO

import glob
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from sklearn.model_selection import train_test_split

import numpy as np


class seed2coco(object):
    def __init__(self, root_dir='/fengyouliang/datasets/seed/train/'):
        self.root_dir = root_dir

        self.image_dir = f'{root_dir}Illegal_adv_images/'
        self.annotation_dir = f'{root_dir}Annotations/'

        self.all_images = glob.glob(f'{self.image_dir}*.jpg')
        self.all_xmls = glob.glob(f'{self.annotation_dir}*.xml')

        self.all_basenames = [Path(item).stem for item in self.all_images]
        self.bbox_index = 0
        self.label_ids = {'window_shielding': 1, 'non_traffic_sign': 2, 'multi_signs': 3}
        self.results = self.load_xml_ann()

    def save_coco(self, save_name=None):
        for mode in ['train', 'val', 'all']:
            instance = self.to_coco_json(mode)
            if save_name is None:
                save_path = f'{self.root_dir}/coco/annotations'
            else:
                save_path = f'{self.root_dir}/coco/annotations/{save_name}'
            os.makedirs(save_path, exist_ok=True)
            json.dump(instance, open(f'{save_path}/{mode}.json', 'w'), ensure_ascii=False, indent=2)
            print(f'{save_path}/{mode}.json')

    def load_xml_ann(self):
        images_train, annotations_train = [], []
        images_val, annotations_val = [], []
        all_images, all_annotations = [], []

        train_name, val_name = train_test_split(self.all_basenames, train_size=0.85)

        for idx, basename in enumerate(train_name):
            image, annotation = self.load_xml(basename, idx)
            images_train.append(image)
            annotations_train.extend(annotation)
        for idx, basename in enumerate(val_name):
            image, annotation = self.load_xml(basename, idx)
            images_val.append(image)
            annotations_val.extend(annotation)
        for idx, basename in enumerate(self.all_basenames):
            image, annotation = self.load_xml(basename, idx)
            all_images.append(image)
            all_annotations.extend(annotation)
        img_ann = {
            'train': [images_train, annotations_train],
            'val': [images_val, annotations_val],
            'all': [all_images, all_annotations],
        }
        return img_ann

    def load_xml(self, basename, image_id):
        xml_path = f'{self.annotation_dir}{basename}.xml'

        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        image = {
            'height': h,
            'width': w,
            'id': image_id,
            'file_name': basename + '.jpg'
        }

        annotation = []

        for obj in root.findall('object'):
            name = obj.find('name').text
            label = self.label_ids[name]

            bnd_box = obj.find('bndbox')
            bbox = [
                float(bnd_box.find('xmin').text),
                float(bnd_box.find('ymin').text),
                float(bnd_box.find('xmax').text),
                float(bnd_box.find('ymax').text),
            ]
            x1, y1, x2, y2 = np.array(bbox) - 1
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            bbox = [x, y, w, h]
            area = w * h
            segmentation = [[x, y, x, y + h, x + w, y + h, x + w, y]]

            box_id = self.bbox_index
            self.bbox_index += 1

            box_item = {
                'image_id': image_id,
                'id': box_id,
                'category_id': label,
                'bbox': bbox,
                'segmentation': segmentation,
                'area': area,
                'iscrowd': 0,
            }
            annotation.append(box_item)
        assert annotation != []
        return image, annotation

    def to_coco_json(self, mode):
        images, annotations = self.results[mode]
        instance = {'info': 'x-ary detection', 'license': ['fengyun']}
        print(f'#images: {len(images)} \t #annontations: {len(annotations)}')
        instance['images'] = images
        instance['annotations'] = annotations
        instance['categories'] = self.get_categories()
        return instance

    def get_categories(self):
        categories = []
        for k, v in self.label_ids.items():
            category = {'id': v, 'name': k}
            categories.append(category)
        return categories


def main():
    for idx in range(1):
        x2coco = seed2coco()
        x2coco.save_coco(save_name=f'fold{idx}')


if __name__ == '__main__':
    main()

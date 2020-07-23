# -*- coding: utf-8 -*-

import json
import os
import os.path as osp

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

np.random.seed(41)


class config:
    data_root = '/fengyouliang/datasets/WHD'


def cvt_csv(whd_df):
    # change dtype
    whd_df[['bbox_xmin', 'bbox_ymin', 'bbox_width', 'bbox_height']] = whd_df['bbox'].str.split(',', expand=True)
    whd_df['bbox_xmin'] = whd_df['bbox_xmin'].str.replace('[', '').astype(float)
    whd_df['bbox_ymin'] = whd_df['bbox_ymin'].str.replace(' ', '').astype(float)
    whd_df['bbox_width'] = whd_df['bbox_width'].str.replace(' ', '').astype(float)
    whd_df['bbox_height'] = whd_df['bbox_height'].str.replace(']', '').astype(float)

    # add xmax, ymax, and area columns for bounding box
    whd_df['bbox_xmax'] = whd_df['bbox_xmin'] + whd_df['bbox_width']
    whd_df['bbox_ymax'] = whd_df['bbox_ymin'] + whd_df['bbox_height']
    whd_df['bbox_area'] = whd_df['bbox_height'] * whd_df['bbox_width']
    return whd_df


class WHD2COCO:

    def __init__(self, kaggle_csv_name, random_seed=None, data_root=config.data_root, min_area_limit=400,
                 max_area_limit=100000,
                 class_agnostic=True, stratified=True, test_size=0.2):
        self.data_root = data_root
        self.kaggle_csv_file = f'{data_root}/kaggle_csv/{kaggle_csv_name}.csv'
        assert osp.isfile(self.kaggle_csv_file) is True, f'{self.kaggle_csv_file} is not found ! \nCheck!'
        self.class_agnostic = class_agnostic
        self.min_area_limit = min_area_limit
        self.max_area_limit = max_area_limit
        self.random_seed = random_seed
        self.stratified = stratified
        self.test_size = test_size

        if self.class_agnostic:
            self.classname_to_id = {'wheat': 0}
        else:
            self.classname_to_id = {
                'usask_1': 0,
                'arvalis_1': 1,
                'inrae_1': 2,
                'ethz_1': 3,
                'arvalis_3': 4,
                'rres_1': 5,
                'arvalis_2': 6,
            }

    def split_train_val(self):
        data_root_path = self.data_root

        original_train = pd.read_csv(self.kaggle_csv_file)
        image_ids = original_train['image_id'].unique()

        if self.stratified:
            image_source = original_train[['image_id', 'source']].drop_duplicates()

            # get lists for image_ids and sources
            image_ids = image_source['image_id'].to_numpy()
            sources = image_source['source'].to_numpy()

            ret = train_test_split(image_ids, sources, test_size=self.test_size, stratify=sources)
            train, val, y_train, y_val = ret
        else:
            train, val = train_test_split(image_ids, test_size=self.test_size, random_state=self.random_seed)

        train_csv = original_train.loc[original_train['image_id'].isin(train)]
        val_csv = original_train.loc[original_train['image_id'].isin(val)]

        train_csv.to_csv(f'{data_root_path}/split_csv/train.csv')
        val_csv.to_csv(f'{data_root_path}/split_csv/val.csv')

    def to_coco(self, mode):
        instance = {'info': 'wheat detection', 'license': ['fengyun']}
        images, annotations = self.load_csv_ann(mode)
        print(f'#images: {len(images)} \t #annontations: {len(annotations)}')
        instance['images'] = images
        instance['annotations'] = annotations
        instance['categories'] = self._get_categories()
        return instance

    def save_coco_json(self, ann_fold_name):
        self.split_train_val()
        save_path = f'{self.data_root}/{ann_fold_name}'

        if not osp.exists(save_path):
            os.makedirs(save_path)
        else:
            assert len(os.listdir(save_path)) == 0, f'{save_path} is not None! \nPlease check'

        for mode in ['train', 'val']:
            save_file = f'{save_path}/{mode}.json'
            instance = self.to_coco(mode)
            json.dump(instance, open(save_file, 'w'), ensure_ascii=False, indent=2)
            print(f'dump done! \n {save_file}')

    def load_csv_ann(self, mode):
        ann_file = f'{self.data_root}/split_csv/{mode}.csv'
        ann_dataframe = pd.read_csv(ann_file)
        ann_dataframe = cvt_csv(ann_dataframe)

        image_ids = ann_dataframe['image_id'].unique()

        images = []
        annotations = []
        box_id = 1

        bar = tqdm(enumerate(image_ids), total=len(image_ids))
        for idx, image_id in bar:
            bar.set_description(f'{image_id}')
            image_idx = idx + 1
            image_set = ann_dataframe[ann_dataframe['image_id'] == image_id]

            image = dict()
            image['height'] = 1024
            image['width'] = 1024
            image['id'] = image_idx
            image['file_name'] = image_id + '.jpg'

            images.append(image)

            bboxes = image_set[['bbox_xmin', 'bbox_ymin', 'bbox_width', 'bbox_height']].values
            labels = image_set[['source']].values
            areas = image_set[['bbox_area']].values

            for bb_idx, (bbox, label, area) in enumerate(zip(bboxes, labels, areas)):
                box_item = dict()
                box_item['bbox'] = list(bbox)
                x, y, w, h = box_item['bbox']
                if self.bbox_filter(w, h):
                    continue
                box_item['segmentation'] = [[x, y, x, y + h, x + w, y + h, x + w, y]]
                box_item['id'] = box_id
                box_id += 1
                box_item['image_id'] = image_idx
                if self.class_agnostic:
                    box_item['category_id'] = self.classname_to_id['wheat']  # == 0
                else:
                    box_item['category_id'] = self.classname_to_id[label[0]]
                box_item['area'] = area[0]
                box_item['iscrowd'] = 0
                annotations.append(box_item)

        return images, annotations

    def _image(self, path):
        image = {}
        print(path)
        img = cv2.imread(self.image_dir + path)
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = self.img_id
        image['file_name'] = path
        return image

    def _get_categories(self):
        categories = []
        for k, v in self.classname_to_id.items():
            category = {'id': v, 'name': k}
            categories.append(category)
        return categories

    def bbox_filter(self, w, h):
        if not self.min_area_limit < w * h < self.max_area_limit:
            return False
        # if w < 10 or h < 10:
        #     return True
        # if w > 512 or h > 512:
        #     return True
        return False


def split_fold(num_fold=5):
    base_name = f'cross_validation'
    for i in range(num_fold):
        fold_name = f'{base_name}/fold_{i}'
        WHD2COCO('train_0618', random_seed=i, class_agnostic=True).save_coco_json(ann_fold_name=fold_name)


def gen_coco_dataset():
    fold_name = 'annotations'
    WHD2COCO('train_0618', random_seed=None, class_agnostic=True, test_size=0.1).save_coco_json(ann_fold_name=fold_name)


if __name__ == '__main__':
    gen_coco_dataset()

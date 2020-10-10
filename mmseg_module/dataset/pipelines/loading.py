import os.path as osp
import random

import mmcv
import numpy as np
from mmseg.datasets import PIPELINES
from mmseg.datasets.pipelines import LoadImageFromFile, LoadAnnotations


@PIPELINES.register_module()
class LoadMosaicImageAndAnnotations(object):
    def __init__(self,
                 mosaic_size,
                 center=0.5,  # mosaic center
                 center_range=0.5,  # mosaic center range
                 to_float32=False,  # img
                 color_type='color',  # img
                 reduce_zero_label=False,  # ann
                 file_client_args=dict(backend='disk'),
                 img_imdecode_backend='cv2',
                 ann_imdecode_backend='pillow'):
        self.mosaic_size = mosaic_size
        self.center = center
        self.center_range = center_range
        # Image Annotations
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.img_file_client = None
        self.ann_file_client = None
        self.img_imdecode_backend = img_imdecode_backend
        self.ann_imdecode_backend = ann_imdecode_backend
        self.reduce_zero_label = reduce_zero_label

    def __call__(self, results):
        if isinstance(results, list) and len(results) == 4:
            return self._load_mosaic_img_ann(results)
        elif isinstance(results, dict):
            return self._load_img_ann(results)
        else:
            raise ValueError(f"results type: {type(results)} should be list or dict")

    def _load_img_ann(self, results):
        # load image
        file_client_img = mmcv.FileClient(**self.file_client_args.copy())

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = file_client_img.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.img_imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        # load annotation
        file_client_ann = mmcv.FileClient(**self.file_client_args.copy())

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = file_client_ann.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.ann_imdecode_backend).squeeze().astype(np.uint8)
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')

        return results

    def _load_mosaic_img_ann(self, results):
        images = []
        semantic_annotations = []

        filenames = []

        for results_item in results:

            # load image
            if self.img_file_client is None:
                self.img_file_client = mmcv.FileClient(**self.file_client_args.copy())

            if results_item.get('img_prefix') is not None:
                filename = osp.join(results_item['img_prefix'],
                                    results_item['img_info']['filename'])
            else:
                filename = results_item['img_info']['filename']
            filenames.append(filename)
            img_bytes = self.img_file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.img_imdecode_backend)
            if self.to_float32:
                img = img.astype(np.float32)
            images.append(img)

            # load seg
            if self.ann_file_client is None:
                self.ann_file_client = mmcv.FileClient(**self.file_client_args.copy())

            if results_item.get('seg_prefix', None) is not None:
                filename = osp.join(results_item['seg_prefix'],
                                    results_item['ann_info']['seg_map'])
            else:
                filename = results_item['ann_info']['seg_map']
            img_bytes = self.ann_file_client.get(filename)

            gt_semantic_seg = mmcv.imfrombytes(
                img_bytes, flag='unchanged',
                backend=self.ann_imdecode_backend).squeeze().astype(np.uint8)

            # reduce zero_label
            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_semantic_seg[gt_semantic_seg == 0] = 255
                gt_semantic_seg = gt_semantic_seg - 1
                gt_semantic_seg[gt_semantic_seg == 254] = 255

            # results['gt_semantic_seg'] = gt_semantic_seg
            # results['seg_fields'].append('gt_semantic_seg')

            semantic_annotations.append(gt_semantic_seg)

        h, w = self.mosaic_size
        low, high = [self.center - self.center_range / 2, self.center + self.center_range / 2]
        xc, yc = [int(random.uniform(size * low, size * high)) for size in self.mosaic_size]  # center x, y

        for i in range(4):

            cur_image = images[i]
            cur_ann = semantic_annotations[i]

            if i == 0:  # bottom right
                result_image = np.full((h, w, 3), 1, dtype=cur_image.dtype)
                result_ann = np.full((h, w), 255, dtype=cur_ann.dtype)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h,
                                                         0), xc, yc  # xmin, ymin, xmax, ymax (large image) mosaic
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (
                            y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image) original
            elif i == 1:  # bottom left
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # top right
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(h, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # top left
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w), min(h, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            result_image[y1a:y2a, x1a:x2a] = cur_image[y1b:y2b, x1b:x2b]  # result_image[ymin:ymax, xmin:xmax]
            result_ann[y1a:y2a, x1a:x2a] = cur_ann[y1b:y2b, x1b:x2b]  # result_image[ymin:ymax, xmin:xmax]

            # vis code
            # import matplotlib.pyplot as plt
            # plt.imshow(cur_image)
            # plt.gca().add_patch(plt.Rectangle((x1b, y1b), (x2b - x1b), (y2b - y1b), fill=False, edgecolor='r', linewidth=3))
            # plt.show()
            #
            # plt.imshow(cur_ann)
            # plt.gca().add_patch(plt.Rectangle((x1b, y1b), (x2b - x1b), (y2b - y1b), fill=False, edgecolor='r', linewidth=3))
            # plt.show()

        # plt.imshow(result_image)
        # plt.show()
        # plt.imshow(result_ann)
        # plt.show()
        # print()

        # processing image format
        mosaic_results = results[0]
        mosaic_results['filename'] = filenames
        mosaic_results['ori_filename'] = mosaic_results['img_info']['filename']
        mosaic_results['img'] = result_image
        mosaic_results['img_shape'] = result_image.shape
        mosaic_results['ori_shape'] = result_image.shape
        # Set initial values for default meta_keys
        mosaic_results['pad_shape'] = result_image.shape
        mosaic_results['scale_factor'] = 1.0
        num_channels = 1 if len(result_image.shape) < 3 else result_image.shape[2]
        mosaic_results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        # processing ann format
        mosaic_results['gt_semantic_seg'] = result_ann
        mosaic_results['seg_fields'].append('gt_semantic_seg')

        return mosaic_results

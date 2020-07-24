import copy
import os.path as osp
import random

import cv2
import mmcv
import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as maskUtils
from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.datasets.builder import PIPELINES
from skimage import img_as_ubyte


@PIPELINES.register_module()
class LoadMosaicImageAndAnnotations(object):
    def __init__(self, image_shape,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 hsv_aug=True,
                 h_gain=0.014,
                 s_gain=0.68,
                 v_gain=0.36,
                 skip_box_w=0,
                 skip_box_h=0,
                 ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain
        self.hsv_aug = hsv_aug
        self.skip_box_w = skip_box_w
        self.skip_box_h = skip_box_h
        self.image_shape = image_shape

    def __call__(self, results):
        if len(results) == 1:
            results = self._load_image_annotations(results, 0)
        if len(results) == 4:
            results = self._load_mosaic_image_and_annotations(results)
        return results

    def _load_mosaic_image_and_annotations(self, results):
        indexes = [0, 1, 2, 3]
        result_boxes = []
        results_c = copy.deepcopy(results)
        results = results[0]
        imsize = self.image_shape[0]
        # print(imsize)
        w, h = self.image_shape[0], self.image_shape[1]
        # s = imsize // 2
        s = imsize
        xc, yc = [int(random.uniform(s * 2 * 0.4, s * 2 * 0.6)) for _ in range(2)]  # center x, y
        # print(f"xc: {xc}, yc: {yc}")

        for i, index in enumerate(indexes):
            result = self._load_image_annotations(results_c, index)
            # print(result.keys())
            image = result['img'].astype(np.float32)
            h, w, c = image.shape
            boxes = result['gt_bboxes']
            # result_masks.append(result['gt_masks'])
            labels = result['gt_labels']

            # labels = labels.reshape(-1, 1)
            # boxs_labels = np.concatenate([boxes, labels], axis=1)

            # TODO: show original image and annotations
            # plt.imshow(image.astype(np.uint8))
            # for box in boxes:
            #     x1, y1, x2, y2 = box
            #     plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='r', linewidth=3))
            # plt.show()

            if i == 0:
                result_image = np.full((s * 2, s * 2, 3), 1, dtype=np.float32)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image) mosaic
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image) original
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            origin_wh = [x2b - x1b, y2b - y1b]
            mosaic_wh = [x2a - x1a, y2a - y1a]
            # print(origin_wh)
            # print(mosaic_wh)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]  # result_image[ymin:ymax, xmin:xmax]

            # print('xmin, ymin, xmax, ymax')
            # print(f"original image xyxy: {[x1b, y1b, x2b, y2b]}")
            # print(f"mosaic image xyxy: {[x1a, y1a, x2a, y2a]}")

            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes
        result_boxes = result_boxes[
            np.where((result_boxes[:, 2] - result_boxes[:, 0]) > self.skip_box_w)]
        result_boxes = result_boxes[
            np.where((result_boxes[:, 3] - result_boxes[:, 1]) > self.skip_box_h)]

        # TODO: show mosaic image and annotations
        # plt.imshow(result_image.astype(np.uint8))
        # for box in result_boxes:
        #     x1, y1, x2, y2 = box
        #     plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='r', linewidth=3))
        # plt.show()

        # results = self._load_image_boxes(results, 0)

        results['ann_info']['bboxes'] = result_boxes
        result_labels = np.zeros(len(result_boxes), dtype=np.long)
        results['ann_info']['labels'] = result_labels

        masks = []
        for box in result_boxes:
            min_x = box[0]
            min_y = box[1]
            max_x = box[2]
            max_y = box[3]
            mask_h = max_y - min_y
            mask_w = max_x - min_x
            masks.append([[min_x, min_y, min_x, min_y + 0.5 * mask_h, min_x, max_y, min_x + 0.5 * mask_w, max_y, max_x,
                           max_y, max_x,
                           max_y - 0.5 * mask_h, max_x, min_y, max_x - 0.5 * mask_w, min_y]])

        results['ann_info']['masks'] = masks

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        result_image = img_as_ubyte(result_image / 255.0)
        if self.hsv_aug:
            augment_hsv(img=result_image, hgain=self.h_gain, sgain=self.s_gain, vgain=self.v_gain)
        results['img'] = result_image
        results['img_shape'] = result_image.shape
        results['ori_shape'] = result_image.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = result_image.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(result_image.shape) < 3 else result_image.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def _load_image_annotations(self, results, k):
        results = results[k]
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)

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
        results['img_fields'] = ['img']

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)

        return results

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        """Private function to load label annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to
        bitmaps.
        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.
        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.
        Args:
            polygons (list[list]): Polygons of one instance.
        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        """Private function to load mask annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        """Private function to load semantic segmentation annotations.
        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        results['gt_semantic_seg'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'image_shape={self.image_shape}, '
        repr_str += f'hsv_aug={self.hsv_aug}, '
        repr_str += f'h_gain={self.h_gain}, '
        repr_str += f's_gain={self.s_gain}, '
        repr_str += f'v_gain={self.v_gain}, '
        repr_str += f'skip_box_w={self.skip_box_w}, '
        repr_str += f'skip_box_h={self.skip_box_h}, '
        repr_str += f'to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}', "
        repr_str += f'with_seg={self.with_seg})'
        repr_str += f'poly2mask={self.poly2mask})'
        repr_str += f'poly2mask={self.file_client_args})'
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

import os.path as osp

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

import numpy as np
from mmcv.utils import print_log


@DATASETS.register_module()
class PCLDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('water', 'traffic', 'architecture', 'cultivated land', 'greening', 'plantation', 'bare soil', 'other')
    # 水体, 交通运输, 建筑, 耕地, 绿化, 人工林, 裸土, 其它
    PALETTE = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [0, 0, 0]]

    def __init__(self, split, **kwargs):
        super(PCLDataset, self).__init__(img_suffix='.tif', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None

    # def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
    #     """Evaluate the dataset.
    #
    #     Args:
    #         results (list): Testing results of the dataset.
    #         metric (str | list[str]): Metrics to be evaluated.
    #         logger (logging.Logger | None | str): Logger used for printing
    #             related information during evaluation. Default: None.
    #
    #     Returns:
    #         dict[str, float]: Default metrics.
    #     """
    #
    #     if not isinstance(metric, str):
    #         assert len(metric) == 1
    #         metric = metric[0]
    #     allowed_metrics = ['mIoU']
    #     if metric not in allowed_metrics:
    #         raise KeyError('metric {} is not supported'.format(metric))
    #
    #     eval_results = {}
    #     num_classes = len(self.CLASSES)
    #
    #     # all_acc, acc, iou = mean_iou(
    #     #     results, gt_seg_maps, num_classes, ignore_index=self.ignore_index)
    #     all_acc, acc, iou = results
    #
    #     summary_str = ''
    #     summary_str += 'per class results:\n'
    #
    #     line_format = '{:<15} {:>10} {:>10}\n'
    #     summary_str += line_format.format('Class', 'IoU', 'Acc')
    #     if self.CLASSES is None:
    #         class_names = tuple(range(num_classes))
    #     else:
    #         class_names = self.CLASSES
    #     for i in range(num_classes):
    #         iou_str = '{:.2f}'.format(iou[i] * 100)
    #         acc_str = '{:.2f}'.format(acc[i] * 100)
    #         summary_str += line_format.format(class_names[i], iou_str, acc_str)
    #     summary_str += 'Summary:\n'
    #     line_format = '{:<15} {:>10} {:>10} {:>10}\n'
    #     summary_str += line_format.format('Scope', 'mIoU', 'mAcc', 'aAcc')
    #
    #     iou_str = '{:.2f}'.format(np.nanmean(iou) * 100)
    #     acc_str = '{:.2f}'.format(np.nanmean(acc) * 100)
    #     all_acc_str = '{:.2f}'.format(all_acc * 100)
    #     summary_str += line_format.format('global', iou_str, acc_str,
    #                                       all_acc_str)
    #     print_log(summary_str, logger)
    #
    #     eval_results['mIoU'] = np.nanmean(iou)
    #     eval_results['mAcc'] = np.nanmean(acc)
    #     eval_results['aAcc'] = all_acc
    #
    #     return eval_results

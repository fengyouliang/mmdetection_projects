import os.path as osp

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class PCLDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('background', 'scratch')

    PALETTE = [[0, 0, 0], [128, 0, 0]]

    def __init__(self, split=None, **kwargs):
        super(PCLDataset, self).__init__(
            img_suffix='.bmp', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir)

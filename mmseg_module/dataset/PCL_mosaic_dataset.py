import os.path as osp

from mmseg.datasets.builder import DATASETS

from mmseg_module.dataset import MosaicCustomDataset


@DATASETS.register_module()
class PCLMosaicDataset(MosaicCustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('water', 'traffic', 'architecture', 'cultivated land', 'greening', 'plantation', 'bare soil', 'other')
    # 水体, 交通运输, 建筑, 耕地, 绿化, 人工林, 裸土, 其它
    PALETTE = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
               [0, 0, 0]]

    def __init__(self, split, **kwargs):
        super(PCLMosaicDataset, self).__init__(img_suffix='.tif', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None

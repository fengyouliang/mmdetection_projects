from collections import OrderedDict

from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16, force_fp32
from mmseg.models import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from torch import nn
from mmcv.cnn import normal_init


@HEADS.register_module()
class UFCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 in_index=(0, 1, 2, 3),
                 lateral_channels=256,
                 **kwargs):
        if isinstance(lateral_channels, int):
            self.lateral_channels = [lateral_channels] * len(in_channels)
        self.lateral_channels = lateral_channels

        self.in_channels = in_channels
        self.in_index = in_index
        super(UFCNHead, self).__init__(in_channels=in_channels, in_index=in_index, **kwargs)

        kernel_size = 3
        lateral_convs = {
            f"conv{i}":
            ConvModule(
                input_channel,
                input_channel,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ) for i, input_channel, in enumerate(self.in_channels)
        }
        self.lateral_convs = nn.Sequential(OrderedDict(lateral_convs))

        self.upconv0 = ConvModule(
            in_channels=2048,
            out_channels=1024,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.upconv1 = ConvModule(
            in_channels=1024,
            out_channels=512,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.upconv2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )

    def init_weights(self):
        """Initialize weights of upconv layer."""
        normal_init(self.upconv2, mean=0, std=0.01)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""

        lateral_outputs = []
        for i in range(len(inputs) - 1, -1, -1):
            x = self.lateral_convs[i](inputs[i])
            lateral_outputs.append(x)
        lateral_outputs[1] = self.upconv0(lateral_outputs[0]) + lateral_outputs[1]
        lateral_outputs[2] = self.upconv1(lateral_outputs[1]) + lateral_outputs[2]
        lateral_outputs[3] = self.upconv2(lateral_outputs[2]) + lateral_outputs[3]
        output = self.cls_seg(lateral_outputs[3])
        return output

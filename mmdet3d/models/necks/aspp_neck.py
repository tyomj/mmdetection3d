# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from mmcv.cnn.resnet import BasicBlock
from mmengine.model import BaseModule
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import OptConfigType


@MODELS.register_module()
class ASPPNeck(BaseModule):
    """ASPPNeck (Atrous Spatial Pyramid Pooling)

    This is a customized version of the ASPP module interface taken from
    MMDetection v3.10.0. The changes are specific to the PillarNeXt
    (https://arxiv.org/abs/2305.04925) implementation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by this module
        kernel_size (int): Size of the convolving kernel.
        dilations (tuple[int]): Dilations of the four branches.
            Default: (1, 6, 12, 18)
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 dilations: Optional[Tuple[int]] = (1, 6, 12, 18),
                 init_cfg: OptConfigType = dict(
                     type='Kaiming', layer='Conv2d')):
        super().__init__(init_cfg)
        self.dilations = dilations
        self.num_aspp = len(dilations)
        self.conv_input = BasicBlock(in_channels, in_channels)
        self.conv1x1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.weight = nn.Parameter(
            torch.randn(in_channels, in_channels, kernel_size, kernel_size))
        self.conv_out = ConvModule(
            in_channels * (2 + self.num_aspp),
            out_channels,
            kernel_size=1,
            stride=1)

    def _forward(self, x: Tensor) -> Tensor:
        """Main forward function of ASPPNeck. In this implementation the
        weights of the dilated convolutions are shared.

        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            Tensor: Output tensor of shape (N, C, H, W)
        """
        x = self.conv_input(x)
        xc_corr = self.conv1x1(x)
        out = []
        for dilation in self.dilations:
            padding = dilation
            out.append(
                F.conv2d(
                    x,
                    self.weight,
                    stride=1,
                    bias=None,
                    padding=padding,
                    dilation=dilation))
        out = torch.cat(out, dim=1)
        x = self.conv_out(torch.cat((x, xc_corr, out), dim=1))
        return x

    def forward(
            self, x: Union[Tensor, Tuple[Tensor],
                           List[Tensor]]) -> Tuple[Tensor]:
        """Forward of ASPPNeck."""
        if isinstance(x, tuple) or isinstance(x, list):
            if len(x) == 1:
                x = x[0]
            else:
                raise ValueError('ASPPNeck only accepts one input tensor')

        if x.requires_grad:
            out = cp.checkpoint(self._forward, x)
        else:
            out = self._forward(x)

        return tuple([out])

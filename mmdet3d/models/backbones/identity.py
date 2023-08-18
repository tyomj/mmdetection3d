# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class IdentityBackbone(nn.Module):

    def __init__(self):
        super(IdentityBackbone, self).__init__()

    def forward(self, x):
        return x

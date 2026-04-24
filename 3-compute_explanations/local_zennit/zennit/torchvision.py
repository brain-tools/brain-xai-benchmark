# This file is part of Zennit
# Copyright (C) 2019-2021 Christopher J. Anders
#
# zennit/torchvision.py
#
# Zennit is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# Zennit is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
# more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library. If not, see <https://www.gnu.org/licenses/>.
'''Specialized Canonizers for models from torchvision.'''
import torch
from torchvision.models.resnet import Bottleneck as ResNetBottleneck, BasicBlock as ResNetBasicBlock

from .canonizers import SequentialMergeBatchNorm, AttributeCanonizer, CompositeCanonizer
from .layer import Sum


class VGGCanonizer(SequentialMergeBatchNorm):
    '''Canonizer for torchvision.models.vgg* type models. This is so far identical to a SequentialMergeBatchNorm'''


class ResNet3DBasicBlockCanonizer(AttributeCanonizer):
    """Canonizer specifically for Hara et al. 3D ResNet BasicBlock blocks."""
    def __init__(self, basicblock_cls):
        """
        Parameters
        ----------
        basicblock_cls : type
            The BasicBlock class from the Hara 3D ResNet implementation.
        """
        self.basicblock_cls = basicblock_cls
        super().__init__(self._attribute_map)

    def _attribute_map(self, name, module):
        """Overload forward + add Sum module for BasicBlock blocks."""
        if isinstance(module, self.basicblock_cls):
            attributes = {
                "forward": self.forward.__get__(module),
                "canonizer_sum": Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        """Modified BasicBlock forward for Hara 3D ResNet, with explicit Sum."""
        identity = x

        # First conv + BN + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)   # Hara: relu1

        # Second conv + BN
        out = self.conv2(out)
        out = self.bn2(out)

        # Downsample identity if needed (can be nn.Sequential or partial)
        if self.downsample is not None:
            identity = self.downsample(x)

        # Stack for Sum module
        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)

        # Final ReLU
        out = self.relu2(out)   # Hara: relu2

        return out

class ResNet3DBottleneckCanonizer(AttributeCanonizer):
    """Canonizer specifically for Hara et al. 3D ResNet Bottleneck blocks."""
    def __init__(self, bottleneck_cls):
        """
        Parameters
        ----------
        bottleneck_cls : type
            The Bottleneck class from the Hara 3D ResNet implementation.
            (e.g. your_module.Bottleneck)
        """
        self.bottleneck_cls = bottleneck_cls
        super().__init__(self._attribute_map)

    def _attribute_map(self, name, module):
        """Overload forward + add Sum module for Bottleneck blocks."""
        if isinstance(module, self.bottleneck_cls):
            attributes = {
                "forward": self.forward.__get__(module),
                "canonizer_sum": Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        """Modified Bottleneck forward for Hara 3D ResNet, with explicit Sum."""
        identity = x

        # First conv + BN + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        # 2D ResNet uses self.relu; Hara uses relu1/2/3
        out = self.relu1(out)

        # Second conv + BN + ReLU
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # Third conv + BN
        out = self.conv3(out)
        out = self.bn3(out)

        # Downsample identity if needed (can be nn.Sequential or partial)
        if self.downsample is not None:
            identity = self.downsample(x)

        # Stack for Sum module (hooks for LRP)
        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)

        # Final ReLU (Hara uses relu3)
        out = self.relu3(out)

        return out

class ResNetBottleneckCanonizer(AttributeCanonizer):
    '''Canonizer specifically for Bottlenecks of torchvision.models.resnet* type models.'''
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.

        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a Bottleneck layer, the appropriate attributes to overload are returned.

        Returns
        -------
        None or dict
            None if `module` is not an instance of Bottleneck, otherwise the appropriate attributes to overload onto
            the module instance.
        '''
        if isinstance(module, ResNetBottleneck):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified Bottleneck forward for ResNet.'''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)

        out = self.relu(out)

        return out


class ResNetBasicBlockCanonizer(AttributeCanonizer):
    '''Canonizer specifically for BasicBlocks of torchvision.models.resnet* type models.'''
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.

        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a BasicBlock layer, the appropriate attributes to overload are returned.

        Returns
        -------
        None or dict
            None if `module` is not an instance of BasicBlock, otherwise the appropriate attributes to overload onto
            the module instance.
        '''
        if isinstance(module, ResNetBasicBlock):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified BasicBlock forward for ResNet.'''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)

        out = self.relu(out)

        return out

class ResNet3DCanonizer(CompositeCanonizer):
    """
    Canonizer for Hara et al. 3D ResNet models.

    Applies:
      - SequentialMergeBatchNorm (Conv3d + BatchNorm3d fusion)
      - ResNet3DBottleneckCanonizer (skip-add via Sum)
      - ResNet3DBasicBlockCanonizer (skip-add via Sum)
    """
    def __init__(self, basicblock_cls, bottleneck_cls):
        canonizers = [
            SequentialMergeBatchNorm(),
            ResNet3DBottleneckCanonizer(bottleneck_cls),
            ResNet3DBasicBlockCanonizer(basicblock_cls),
        ]
        super().__init__(canonizers)

class ResNetCanonizer(CompositeCanonizer):
    '''Canonizer for torchvision.models.resnet* type models. This applies SequentialMergeBatchNorm, as well as
    add a Sum module to the Bottleneck modules and overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''
    def __init__(self):
        super().__init__((
            SequentialMergeBatchNorm(),
            ResNetBottleneckCanonizer(),
            ResNetBasicBlockCanonizer(),
        ))

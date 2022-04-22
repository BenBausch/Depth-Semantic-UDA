# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
from utils.utils import info_gpu_memory

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


class ResnetEncoder(nn.Module):
    """
    Pytorch module for a resnet encoder. Wrapper for the Resnet implementation of torchvision
    """

    def __init__(self, num_layers, pretrained, num_input_images=1, wanted_scales=[1, 2, 3, 4], num_channels_input=3):
        """
        Creates default torchvision resnet model and applies wanted changes.
        :param num_layers: which resnet to use e.g. 101
        :param pretrained: if weights pretrained on imagenet should be used
        :param num_input_images: how many input images the resnet takes (weights of first convolution will
        be copied that often)
        :param wanted_scales: ordered list of scales at which we want the features. Resolution of the features:
        height_scale = height_img / (2 ** scale), width_scale = width_in / (2 ** scale)
        """
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        wanted_scales.sort()  # sort list do that the features are returned in correct order during forward pass
        self.wanted_scales = wanted_scales

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images, num_channels_input)
        else:
            self.encoder = resnets[num_layers](pretrained)

        # Torchvisions resnet implements 4 layer blocks with 4 different channel values [64, 128, 256, 512]
        # each layer block halves the resolution of the input e.g an input of 1080x1920 has resolution 540x960 after
        # first block
        self.scales_to_layers = nn.ModuleDict({'1': self.encoder.layer1,
                                               '2': self.encoder.layer2,
                                               '3': self.encoder.layer3,
                                               '4': self.encoder.layer4})

        if num_layers > 34:
            # for large resnets increase the channels
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        features = []
        x = (input_image - 0.45) / 0.225  # fixme maybe this needs fix
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))
        for i in self.wanted_scales:
            if i == 1:
                # apply max pooling to the first layer block
                x = self.encoder.maxpool(features[-1])
            else:
                x = features[-1]
            features.append(self.scales_to_layers[str(i)](x))
        return features

    def get_channels_of_forward_features(self):
        channels = [64]
        for i in self.wanted_scales:
            channels.append(self.num_ch_enc[i])
        return channels


class ResNetMultiImageInput(models.ResNet):
    """
    Constructs a torchvision resnet model with varying number of input features in constructor.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_input_images=1, num_channels_input=3):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * num_channels_input, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1, num_channels_input=3):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks,
                                  num_input_images=num_input_images,
                                  num_channels_input=num_channels_input)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        if num_channels_input > 3:
            # use r channel weights for all addtion dimension to the rgb image for example rgbd
            additional_channels = num_channels_input - 3

            rgb_extended_weights = [loaded['conv1.weight']] + \
                                   additional_channels * [loaded['conv1.weight'][:, 0, :, :].unsqueeze(1)]
            rgb_extended_weights = [torch.cat(rgb_extended_weights, 1)]
        else:
            rgb_extended_weights = [loaded['conv1.weight']]

        loaded['conv1.weight'] = torch.cat(
            rgb_extended_weights * num_input_images, 1) / num_input_images

        model.load_state_dict(loaded)
    return model


if __name__ == '__main__':
    a = ResnetEncoder(18, pretrained=True, num_input_images=2).double()
    print(a)

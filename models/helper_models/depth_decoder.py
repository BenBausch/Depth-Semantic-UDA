# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

from collections import OrderedDict
from models.helper_models.layers import *


class DepthDecoderMONODEPTH2(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, upsample_mode='nearest'):
        super(DepthDecoderMONODEPTH2, self).__init__()

        self.upsample_mode = upsample_mode
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips

        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.ordered_layers = []

        # decoder
        self.convs = nn.ModuleDict()
        for i in range(4, -1, -1):

            # upconv_0: convolution over the previous layer
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f"upconv_{i}_0"] = ConvBlock(num_ch_in, num_ch_out)
            self.ordered_layers.append(f"upconv_{i}_0")

            # upconv_1: convolution over the previous layer plus the feature of encoder at the same scale as previous
            # layer output
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]

            self.convs[f"upconv_{i}_1"] = ConvBlock(num_ch_in, num_ch_out)
            self.ordered_layers.append(f"upconv_{i}_1")

            if i in self.scales:
                self.convs[f"dispconv_{i}"] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)
                self.ordered_layers.append(f"dispconv_{i}")

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}

        # Initial previous layer is the
        x = input_features[-1]
        for i in range(4, -1, -1):

            # Convolution of the previous layer
            x = self.convs[f"upconv_{i}_0"](x)
            x = [upsample(x, self.upsample_mode)]

            if self.use_skips and i > 0:
                # join the previous layer and same scale latent features
                x += [input_features[i - 1]]

            x = torch.cat(x, dim=1)
            # convolve over concatenated input
            x = self.convs[f"upconv_{i}_1"](x)
            # if scale shall be part of output convolve to get 1 channel output dept
            if i in self.scales:
                outputs[("disp", i)] = self.sigmoid(self.convs[f"dispconv_{i}"](x))

        return outputs

    def __str__(self):
        description = ''
        for i in self.ordered_layers:
            description += str(self.convs[i])
            if len(i) == 3 and i[2] == 0:
                description += f' --> {self.upsample_mode} upsampling'
            description += '\n'
        return description


class DepthDecoderDADA(nn.Module):
    """
    Modified Code based on
    https://github.com/valeoai/DADA/blob/d657af66ee3e18a052f3fba6c34707e46ed6ea96/dada/model/deeplabv2_depth.py
    """

    def __init__(self, in_channels):
        super(DepthDecoderDADA, self).__init__()
        self.enc4_1 = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1, padding=0, bias=True)
        self.enc4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc4_3 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.enc4_1.weight.data.normal_(0, 0.01)
        self.enc4_2.weight.data.normal_(0, 0.01)
        self.enc4_3.weight.data.normal_(0, 0.01)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        x4_enc = self.enc4_1(input_features)
        x4_enc = self.relu(x4_enc)
        x4_enc = self.enc4_2(x4_enc)
        x4_enc = self.relu(x4_enc)
        x4_enc = self.enc4_3(x4_enc)
        raw_sigmoid = self.sigmoid(torch.mean(x4_enc, dim=1, keepdim=True))
        return x4_enc, raw_sigmoid

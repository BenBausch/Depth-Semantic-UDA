# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from train.layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, upsample_mode='nearest'):
        super(DepthDecoder, self).__init__()

        self.upsample_mode = upsample_mode
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips

        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.ordered_layers = []

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):

            # upconv_0: convolution over the previous layer
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            self.ordered_layers.append(("upconv", i, 0))

            # upconv_1: convolution over the previous layer plus the feature of encoder at the same scale as previous
            # layer output
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]

            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            self.ordered_layers.append(("upconv", i, 1))

            if i in self.scales:
                self.convs[("dispconv", i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)
                self.ordered_layers.append(("dispconv", i))

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # Initial previous layer is the
        x = input_features[-1]
        for i in range(4, -1, -1):

            # Convolution of the previous layer
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x, self.upsample_mode)]

            if self.use_skips and i > 0:
                # join the previous layer and same scale latent features
                x += [input_features[i - 1]]

            x = torch.cat(x, dim=1)
            # convolve over concatenated input
            x = self.convs[("upconv", i, 1)](x)
            # if scale shall be part of output convolve to get 1 channel output dept
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
        return self.outputs

    def __str__(self):
        description = ''
        for i in self.ordered_layers:
            description += str(self.convs[i])
            if len(i) == 3 and i[2] == 0:
                description += f' --> {self.upsample_mode} upsampling'
            description += '\n'
        return description

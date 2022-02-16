# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

from collections import OrderedDict
from models.helper_models.layers import *
from utils.utils import info_gpu_memory


class SemanticDecoderGUDA(nn.Module):
    def __init__(self, num_ch_enc, num_classes, use_skips=True,
                 upsample_mode='nearest', num_ch_dec=[16, 32, 64, 128, 256]):
        super(SemanticDecoderGUDA, self).__init__()

        self.upsample_mode = upsample_mode
        self.use_skips = use_skips

        self.num_output_channels = num_classes
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array(num_ch_dec)

        self.scale_factors = [1, 2, 4, 8]
        self.inputs_to_last_layer = [3, 2, 1, 0]

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

        # final convolution for semantic segmentation
        channels_of_last_4_scales_added = np.sum(self.num_ch_dec[:-1])
        self.convs["final_conv"] = Conv3x3(channels_of_last_4_scales_added, self.num_output_channels)
        self.ordered_layers.append("final_conv")

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):

        outputs = []

        # Initial previous layer is the
        x = input_features[-1]
        for i in range(4, -1, -1):

            # Convolution of the previous layer
            x = self.convs[f"upconv_{i}_0"](x)
            x = [upsample(x, self.upsample_mode)]

            if self.use_skips and i > 0:
                # join the previous layer and same scale latent features
                x = x + [input_features[i - 1]]

            x = torch.cat(x, dim=1)
            # convolve over concatenated input
            x = self.convs[f"upconv_{i}_1"](x)

            if i in self.inputs_to_last_layer:
                # concat all output, but not the first convolution output
                # x will be upsampled to the size of the image before the resnet encoder
                outputs.append(upsample(x, self.upsample_mode, scale_factor=self.scale_factors[i]))

        x = torch.cat(outputs, 1)
        x = self.convs["final_conv"](x)
        return x

    def __str__(self):
        description = ''
        input_layer_ids = []
        for i, lay in enumerate(self.ordered_layers):
            description += f'({i}): ' + str(self.convs[lay])
            if len(lay) == 3 and lay[2] == 0:
                description += f' --> {self.upsample_mode} upsampling'
            if len(lay) == 3 and lay[2] == 1 and lay[1] in self.inputs_to_last_layer:
                input_layer_ids.append(i)
            if lay[0] == "final_conv":
                description += f' with Input == Output from layers '
                for lay_id in input_layer_ids:
                    description += f'#{lay_id}, '
                description = description[:-2]
            description += '\n'
        return description


class SemanticDecoderADVENT(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(SemanticDecoderADVENT, self).__init__()
        self.dec4 = nn.Conv2d(128, inplanes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dec4.weight.data.normal_(0, 0.01)
        self.relu = nn.ReLU(inplace=True)

        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding,
                          dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, encoder_features, depth_features):
        x = self.dec4(depth_features)
        x = self.relu(x)
        x = encoder_features * x
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out
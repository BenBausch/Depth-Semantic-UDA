# Own Packages
from models.helper_models.layers import *


class MotionDecoder(nn.Module):
    """
    Motion decoder inspired by 'Unsupervised Monocular Depth Learning in Dynamic Scenes'
    """
    def __init__(self, num_ch_enc, upsample_mode='bilinear', use_skips=True, auto_mask=True):
        super(MotionDecoder, self).__init__()
        self.auto_mask = auto_mask

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = [32, 64, 128, 256]

        self.upsample_mode = upsample_mode
        self.use_skips = use_skips

        self.ordered_layers = []

        # decoder
        self.convs = nn.ModuleDict()
        for i in range(3, -1, -1):

            # upconv_0: convolution over the previous layer
            num_ch_in = self.num_ch_enc[-1] if i == 3 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f"upconv_{i}_0"] = ConvBlock(num_ch_in, num_ch_out)
            self.ordered_layers.append(f"upconv_{i}_0")

            # upconv_1: convolution over the previous layer plus the feature of encoder at the same scale as previous
            # layer output
            if i > 0:
                num_ch_in = self.num_ch_dec[i]
                if self.use_skips:
                    num_ch_in += self.num_ch_enc[i - 1]
                num_ch_out = self.num_ch_dec[i]

                self.convs[f"upconv_{i}_1"] = ConvBlock(num_ch_in, num_ch_out)
                self.ordered_layers.append(f"upconv_{i}_1")

        self.convs[f"translation_map"] = Conv3x3(self.num_ch_dec[i], 3)
        self.ordered_layers.append(f"translation_map")

        self.sigmoid = nn.Sigmoid()

    def _mask(self, x):
        """Copied from https://github.com/chamorajg/pytorch_depth_and_motion_planning/blob/c688338d9d8d8b7dad5722a5eeb0ed8b393a82a5/object_motion_net.py"""
        sq_x = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        mean_sq_x = torch.mean(sq_x, dim=(0, 2, 3))
        mask_x = (sq_x > mean_sq_x).type(x.dtype)
        x = x * mask_x
        return x

    def forward(self, input_features):
        # Initial previous layer is the
        x = input_features[-1]
        for i in range(3, -1, -1):

            # Convolution of the previous layer
            x = self.convs[f"upconv_{i}_0"](x)
            x = [upsample(x, self.upsample_mode)]

            if i > 0:
                if self.use_skips:
                    # join the previous layer and same scale latent features
                    x += [input_features[i - 1]]

                x = torch.cat(x, dim=1)
                # convolve over concatenated input
                x = self.convs[f"upconv_{i}_1"](x)

        translation_map = 0.001 * self.convs[f"translation_map"](x[0])

        if self.auto_mask:
            translation_map = self._mask(translation_map)

        return translation_map

    def __str__(self):
        description = ''
        for i in self.ordered_layers:
            description += str(self.convs[i])
            if len(i) == 3 and i[2] == 0:
                description += f' --> {self.upsample_mode} upsampling'
            description += '\n'
        return description

import torch
import torch.nn as nn
import torch.nn.functional as F


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    # Source: https://github.com/nianticlabs/monodepth2/blob/master/layers.py
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    # Source: https://github.com/Wallacoloo/printipi (adapted by Monodepth2)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4), device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def get_translation_matrix(translation_vector):
    """
    Source: https://github.com/nianticlabs/monodepth2/blob/master/layers.py
    Convert a translation vector into a 4x4 transformation matrixF
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4, device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def disp_to_depth(disp, min_depth=0.1, max_depth=100):
    """
    # Source: https://github.com/nianticlabs/monodepth2/blob/master/layers.py
    Convert network's sigmoid output into depth prediction.
    The formula for this conversion is given in the 'additional considerations'
    section of the paper'.

    examples: disp = 1 # close objects
    min_disp = 1 / 100 = 0.01
    max_disp = 1 / 0.1 = 10
    scaled_disp = 10
    depth = 1 / 10 = 0.1

    disp = 0 # far objects
    min_disp = 1 / 100 = 0.01
    max_disp = 1 / 0.1 = 10
    scaled_disp = 0.01
    depth = 1 / 10 = 100

    --> depth
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp

    return scaled_disp, depth


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

    def __repr__(self):
        return str(self.conv) + f' --> ELU non-linearity'


class Conv3x3(nn.Module):
    """
    Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

    def __repr__(self):
        return f'ConvLayer 3x3 with {self.in_channels} in_channels and {self.out_channels} out_channels'


def upsample(x, mode='nearest', scale_factor=2):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=True)
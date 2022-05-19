import torch


def normalize_motion_map(res_motion_map, motion_map):
    """Normalizes a residual motion map by the motion map's norm.
    Copied from https://github.com/chamorajg/pytorch_depth_and_motion_planning/blob/c688338d9d8d8b7dad5722a5eeb0ed8b393a82a5/regularizers.py#L30"""
    norm = torch.mean(
        torch.square(motion_map), dim=[1, 2, 3], keepdim=True) * 3.0
    return res_motion_map / torch.sqrt(norm + 1e-12)

# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
import torch.nn.functional as F

def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    Source: https://github.com/nianticlabs/monodepth2/blob/master/layers.py
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return {"de/abs_rel": abs_rel, "de/sq_rel": sq_rel, "de/rms": rmse, \
            "de/log_rms": rmse_log, "da/a1": a1, "da/a2": a2, "da/a3":a3}

def compute_depth_losses(use_gt_scale, use_garg_crop, depth_gt, depth_pred, gt_size, depth_ranges):
    """Compute depth metrics, to allow monitoring during training
    Adapted from: https://github.com/nianticlabs/monodepth2/blob/master/trainer.py
    This isn't particularly accurate as it averages over the entire batch,
    so is only used to give an indication of validation performance
    """
    min_depth = depth_ranges[0]
    max_depth = depth_ranges[1]

    # Die Interpolation ist hier an sich nicht mehr nötig, da alle Bilder vom DataLoader im vornherein auf die
    # spezifizierte Größe angepasst werden
    depth_pred = torch.clamp(F.interpolate(
        depth_pred, gt_size, mode="bilinear", align_corners=False), min_depth, max_depth)
    depth_pred = depth_pred.detach()

    mask = depth_gt > 0

    # garg/eigen crop
    if use_garg_crop:
        _, _, gt_height, gt_width = depth_gt.shape
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, int(0.4080*gt_height):int(0.9891*gt_height), int(0.0354*gt_width):int(0.9638*gt_width)] = 1
        mask *= crop_mask

    depth_gt = depth_gt[mask]
    depth_pred = depth_pred[mask]

    # Normalize the depth predictions only if the training was fully unsupervised as the scales cannot be estimated
    # then
    ratios = []
    if(use_gt_scale):
        ratio = torch.median(depth_gt) / torch.median(depth_pred)
        ratios.append(ratio)
        depth_pred *= ratio

    depth_pred = torch.clamp(depth_pred, min=min_depth, max=max_depth)

    depth_errors = compute_depth_errors(depth_gt, depth_pred)

    # Move to cpu
    losses = {}
    for metric, value in depth_errors.items():
        losses[metric] = value.cpu()

    return ratios, losses


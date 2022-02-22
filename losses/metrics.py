import numpy as np
from wandb.wandb_torch import torch
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt


class MIoU:
    """
    Mean Intersection Over Union
    Code based on https://www.kaggle.com/ligtfeather/semantic-segmentation-is-easy-with-pytorch and adapted to calculate
    MIoU over whole dataset.
    """

    def __init__(self, num_classes, ignore_classes=None, ignore_index=250):
        self.classes = torch.tensor([i for i in range(num_classes)])
        self.intersection = torch.tensor([0.0 for i in range(num_classes)])
        self.union = torch.tensor([0.0 for i in range(num_classes)])
        self.ignore_index = ignore_index

        if ignore_classes is not None:
            assert len(ignore_classes) == len(self.classes)
            self.classes = self.classes[ignore_classes]
        print(len(self.classes))

    def update(self, mask_pred, mask_gt):
        """
        Call this function for each batch. Updates the number of true positives (self.intersection) and
        the sum true positives + false positives + false negatives (self.union) per class.
        :param mask_pred: softmax of network mask prediction [batch_size, num_classes, image_height, image_width]
        :param mask_gt: mask ground truth [batch_size, 1, image_height, image_width]
        """
        with torch.no_grad():
            shape = (mask_pred.shape[2], mask_pred.shape[3])
            pred_mask = torch.argmax(mask_pred, dim=1)
            pred_mask = pred_mask.view(-1)
            mask = mask_gt.view(-1)
            not_ingore_index = mask != self.ignore_index

            for cls in self.classes:
                true_class = pred_mask == cls
                true_label = mask == cls
                # consider only the pixels that are not labeled as ignore index, pixels labeled ignore_index are
                # impossible to predict correctly!
                true_class = torch.logical_and(true_class, not_ingore_index)

                if true_label.long().sum().item() == 0:
                    pass
                else:
                    self.intersection[cls] += torch.logical_and(true_class, true_label).sum().float().item()
                    self.union[cls] += torch.logical_or(true_class, true_label).sum().float().item()

    def get_miou(self):
        """
        Returns the mean intersection over union over all classes and images (on which update has been called).
        Additionally the mean iou per class, which averages over all images per class, is returned.
        """
        iou_per_class = self.intersection / self.union
        return torch.nansum(iou_per_class) / len(self.classes), iou_per_class


class DepthEvaluator:
    def __init__(self, use_garg_crop=False):
        self.use_garg_crop = use_garg_crop

    def depth_losses_as_string(self, depth_gt, depth_pred):
        depth_losses = self.compute_depth_losses(depth_gt=depth_gt, depth_pred=depth_pred)
        loss_string = 'Losses: \n'
        for metric, value in depth_losses.items():
            loss_string += f'{metric}: {value} \n'
        return loss_string

    def compute_depth_losses(self, depth_gt, depth_pred):
        """Compute depth metrics, to allow monitoring during training
        Adapted from: https://github.com/nianticlabs/monodepth2/blob/master/trainer.py
        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = depth_pred.detach()

        mask = depth_gt > 0

        # garg/eigen crop
        if self.use_garg_crop:
            _, _, gt_height, gt_width = depth_gt.shape
            crop_mask = torch.zeros_like(mask)
            crop_mask[:, :, int(0.4080 * gt_height):int(0.9891 * gt_height),
            int(0.0354 * gt_width):int(0.9638 * gt_width)] = 1
            mask *= crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]

        depth_errors = self.compute_depth_errors(depth_gt, depth_pred)

        # Move to cpu
        losses = {}
        for metric, value in depth_errors.items():
            losses[metric] = value.cpu()

        return losses

    @staticmethod
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
                "de/log_rms": rmse_log, "da/a1": a1, "da/a2": a2, "da/a3": a3}

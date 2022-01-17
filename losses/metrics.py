import numpy as np
from wandb.wandb_torch import torch
import torch.nn.functional as F
import torch


class MIoU:
    """
    Mean Intersection Over Union
    Code based on https://www.kaggle.com/ligtfeather/semantic-segmentation-is-easy-with-pytorch and adapted to calculate
    MIoU over whole dataset.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.intersection = torch.tensor([0.0 for i in range(self.num_classes)])
        self.union = torch.tensor([0.0 for i in range(self.num_classes)])

    def update(self, mask_pred, mask_gt):
        """
        Call this function for each batch. Updates the number of true positives (self.intersection) and
        the sum true positives + false positives + false negatives (self.union) per class.
        :param mask_pred: softmax of network mask prediction [batch_size, num_classes, image_height, image_width]
        :param mask_gt: mask ground truth [batch_size, 1, image_height, image_width]
        """
        with torch.no_grad():
            pred_mask = torch.argmax(mask_pred, dim=1)
            pred_mask = pred_mask.view(-1)
            mask = mask_gt.view(-1)

            for cls in range(0, self.num_classes):
                true_class = pred_mask == cls
                true_label = mask == cls

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
        return torch.nanmean(iou_per_class), iou_per_class

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
            print(self.classes)


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

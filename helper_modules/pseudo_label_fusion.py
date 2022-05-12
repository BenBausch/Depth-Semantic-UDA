import torch.nn.functional as F
import torch

from utils.constans import IGNORE_INDEX_SEMANTIC


def fuse_pseudo_labels_with_gorund_truth(pseudo_label_prediction, ground_turth_labels, probability_threshold=0.99):
    """
    Creates a Long Tensor Pseudo Label/Ground Truth target tensor. In the UDA setup, pseudo labels come from the target
    image prediction and ground truth labels come from the source domain instances.
    :param pseudo_label_prediction: (non-softmax) tensor of the shape [batch_size, num_classes, img_height, img_width]
    :param ground_turth_labels: Long tensor of the shape [batch_size, img_height, img_width]
    :param probability_threshold: threshold between 0 and 1 to select confident pseudo labels from prediction
    :return: Long Tensor of the shape [batch_size, img_height, img_width]
    """
    soft_pseudo_label = F.softmax(pseudo_label_prediction, dim=1)
    pseudo_labels = torch.argmax(soft_pseudo_label, dim=1)
    pseudo_label_probs = torch.max(soft_pseudo_label, dim=1).values
    pseudo_labels[pseudo_label_probs < probability_threshold] = IGNORE_INDEX_SEMANTIC
    mask = ground_turth_labels != IGNORE_INDEX_SEMANTIC
    pseudo_labels[mask] = ground_turth_labels[mask]

    return pseudo_labels

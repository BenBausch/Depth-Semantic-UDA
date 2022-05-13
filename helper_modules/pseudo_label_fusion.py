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
    pseudo_labels = torch.argmax(pseudo_label_prediction, dim=1)
    pseudo_label_probs = torch.max(pseudo_label_prediction, dim=1).values
    pseudo_labels[pseudo_label_probs <= probability_threshold] = IGNORE_INDEX_SEMANTIC
    mask = ground_turth_labels != IGNORE_INDEX_SEMANTIC
    pseudo_labels[mask] = ground_turth_labels[mask]

    return pseudo_labels

if __name__ == "__main__":
    pred = torch.tensor([[[[0.1, 0., 0.2, 0., 0.],
                           [0.2, 0.2, 0.3, 0., 0.3],
                           [1.0, 0.05, 0.1, 0., 1.]],

                          [[0.7, 0., 0.2, 0.9, 0.6],
                           [0.7, 0., 0.1, 1., 0.7],
                           [0., 0.9, 0.9, 1., 0.]],

                          [[0.2, 1.0, 0.6, 0.1, 0.4],
                           [0.1, 0.8, 0.6, 0., 0.],
                           [0., 0.05, 0.0, 0., 0.]]]])

    ground_turth_labels = torch.LongTensor([[[1, 250, 250, 250, 0],
                                             [1, 250, 250, 2, 0],
                                             [250, 250, 250, 250, 250]]])

    print(fuse_pseudo_labels_with_gorund_truth(pred, ground_turth_labels, probability_threshold=0.89))

from utils.constans import *


def get_cross_entropy_weights(num_classes, dataset_name):
    """
    Returns a tensor of class weights inversely proportional to the % of pixels for each class in the given dataset.
    :param num_classes: number of classes in the dataset
    :param dataset_name: name of the dataset
    :return: tensor of length num_classes with one weight for each class
    """
    if dataset_name == 'cityscapes_semantic':
        if num_classes == 19:
            return CITYSCAPES_19_CLASSES_WEIGHTS
        elif num_classes == 16:
            return CITYSCAPES_16_CLASSES_WEIGHTS
        else:
            raise ValueError(f"No class weights for {dataset_name} with {num_classes} classes defined!")
    elif dataset_name == 'synthia_rand_cityscapes':
        if num_classes == 19:
            return SYNTHIA_19_CLASSES_WEIGHTS
        elif num_classes == 16:
            return SYNTHIA_16_CLASSES_WEIGHTS
        else:
            raise ValueError(f"No class weights for {dataset_name} with {num_classes} classes defined!")
    else:
        raise ValueError(f"No class weights for {dataset_name} defined!")
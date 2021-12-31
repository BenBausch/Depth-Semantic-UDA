import numpy as np
import torch
import PIL.Image as pil
from torchvision import transforms as tf
import cv2


class PrepareForNet(object):
    def __init__(self):
        self.to_tensor = tf.transforms.ToTensor()

    def __call__(self, sample):

        return self.to_tensor(sample).double()


class NormalizeRGB(object):
    def __init__(self, mean, do_norm):
        self.mean = mean
        self.do_norm = do_norm

    def __call__(self, sample):
        sample = sample.astype(np.float64)
        sample -= self.mean
        if self.do_norm:
            sample = sample / 255.0
        return sample


# Wrapper for the PIL resize method to be able to use it in the Compose structure later
class PILResize(object):
    def __init__(self, size, interpolation, box=None, reducing_gap=None):
        self.size = size
        self.interpolation = interpolation
        self.box = box
        self.reducing_gap = reducing_gap

    def __call__(self, sample):
        return sample.resize(self.size, resample=self.interpolation, box=self.box, reducing_gap=self.reducing_gap)


# Wrapper for the cv2 resize method to be able to use it in the Compose structure later
class CV2Resize(object):
    def __init__(self, size, interpolation, box=None, reducing_gap=None):
        self.size = size
        self.interpolation = interpolation
        self.box = box
        self.reducing_gap = reducing_gap

    def __call__(self, sample):
        return cv2.resize(sample, dsize=self.size, interpolation=self.interpolation)


# Wrapper for the PIL transpose method to be able to use it in the Compose structure later
class PILHorizontalFlip(object):
    def __init__(self, do_flip):
        self.do_flip = do_flip

    def __call__(self, sample):
        if self.do_flip:
            return sample.transpose(pil.FLIP_LEFT_RIGHT)
        else:
            return sample


# Wrapper for the CV2 flip method to be able to use it in the Compose structure later
class CV2HorizontalFlip(object):
    def __init__(self, do_flip):
        self.do_flip = do_flip

    def __call__(self, sample):
        if self.do_flip:
            return cv2.flip(sample, 1)  # 1 = flip around y-axis
        else:
            return sample


# Wrapper for the PIL transpose method to be able to use it in the Compose structure later
class ColorAug(object):
    def __init__(self, do_aug, aug_params):
        self.do_aug = do_aug
        if do_aug:
            self.color_aug_obj = tf.ColorJitter(
                (1 - aug_params["brightness_jitter"], 1 + aug_params["brightness_jitter"]),
                (1 - aug_params["contrast_jitter"], 1 + aug_params["contrast_jitter"]),
                (1 - aug_params["saturation_jitter"], 1 + aug_params["saturation_jitter"]),
                (-aug_params["hue_jitter"], +aug_params["hue_jitter"]))

    def __call__(self, sample):
        if self.do_aug:
            return self.color_aug_obj(sample)
        else:
            return sample


# ----------------------------------------------------------------------------------------
# --------------------------Dataset specific transformations------------------------------
# ----------------------------------------------------------------------------------------

# -------------------------------Synthia Rand Cityscapes----------------------------------
class Syntia_To_Cityscapes_Encoding(object):
    def __init__(self, s_to_c_mapping, valid_classes, void_classes, ignore_index):
        self.s_to_c = s_to_c_mapping
        self.valid_classes = valid_classes
        self.void_classes = void_classes
        self.ignore_index = ignore_index

    def __call__(self, label):
        label_copy = np.copy(label)
        for i in self.void_classes:
            label_copy[label == i] = self.ignore_index
        for i in self.valid_classes:
            label_copy[label == i] = self.s_to_c[i]
        return label_copy


class TransformToDepthSynthia(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return np.array(sample).astype(np.float32) / 100 # fix me find out number by which to divide


# ------------------------------------------GTA5------------------------------------------
class ToUint8Array(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return np.asarray(sample, dtype=np.uint8)


class EncodeSegmentation(object):
    def __init__(self, void_classes, valid_classes, class_map, ignore_index):
        self.void_classes = void_classes
        self.valid_classes = valid_classes
        self.class_map = class_map
        self.ignore_index = ignore_index

    def encode_segmap(self, lbl):
        """
            Copied from https://github.com/RogerZhangzz/CAG_UDA/blob/master/data/gta5_dataset.py
            Removed potential overwriting bug.
        """
        lbl_copy = np.copy(lbl)
        for _i in self.void_classes:
            lbl_copy[lbl == _i] = self.ignore_index
        for _i in self.valid_classes:
            lbl_copy[lbl == _i] = self.class_map[_i]
        return lbl_copy

    def __call__(self, sample):
        return self.encode_segmap(sample)


# ------------------------------------------Kitti-----------------------------------------
class TransformToDepthKitti(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return np.array(sample).astype(np.float32) / 256


# ------------------------------------------Kalimu----------------------------------------
class TransformToDepthKalimu(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return np.array(sample).astype(np.float32) / 256
import numpy as np
import torch
import PIL.Image as pil
from torchvision import transforms as tf

class TransformToDepthKitti(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return np.array(sample).astype(np.float32) / 256


class TransformToDepthKalimu(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return np.array(sample).astype(np.float32) / 256


class PrepareForNet(object):
    def __init__(self):
        self.to_tensor = tf.transforms.ToTensor()

    def __call__(self, sample):
        return self.to_tensor(sample)


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
        """
        for _i in self.void_classes:
            lbl[lbl == _i] = self.ignore_index
        for _i in self.valid_classes:
            lbl[lbl == _i] = self.class_map[_i]
        return lbl

    def __call__(self, sample):
        return self.encode_segmap(sample)


class NormalizeRGB(object):
    def __init__(self, mean, do_norm):
        self.mean = mean
        self.do_norm = do_norm

    def __call__(self, sample):
        sample = sample.astype(np.float64)
        sample -= self.mean
        if self.do_norm:
            sample = sample.astype(float) / 255.0
        return sample


# Wrapper for the PIL resize method to be able to use it in the Compose structure later
class Resize(object):
    def __init__(self, size, interpolation, box=None, reducing_gap=None):
        self.size = size
        self.interpolation = interpolation
        self.box = box
        self.reducing_gap = reducing_gap

    def __call__(self, sample):
        return sample.resize(self.size, resample=self.interpolation, box=self.box, reducing_gap=self.reducing_gap)


# Wrapper for the PIL transpose method to be able to use it in the Compose structure later
class HorizontalFlip(object):
    def __init__(self, do_flip):
        self.do_flip = do_flip

    def __call__(self, sample):
        if self.do_flip:
            return sample.transpose(pil.FLIP_LEFT_RIGHT)
        else:
            return sample


# Wrapper for the PIL transpose method to be able to use it in the Compose structure later
class ColorAug(object):
    def __init__(self, do_aug, aug_params):
        self.do_aug = do_aug
        if do_aug:
            self.color_aug_obj = tf.transforms.ColorJitter.get_params(
                (1 - aug_params["brightness_jitter"], 1 + aug_params["brightness_jitter"]),
                (1 - aug_params["contrast_jitter"], 1 + aug_params["contrast_jitter"]),
                (1 - aug_params["saturation_jitter"], 1 + aug_params["saturation_jitter"]),
                (-aug_params["hue_jitter"], +aug_params["hue_jitter"]))

    def __call__(self, sample):
        if self.do_aug:
            return self.color_aug_obj(sample)
        else:
            return sample

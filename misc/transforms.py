import numpy
import numpy as np
import torch
import PIL.Image as pil
import torchvision.transforms
from torchvision import transforms as tf
import cv2

# own modules
from utils.decorators import deprecated


class PrepareForNet(object):
    """
    Turns data into a double tensor and optionally normalizes the data by subtracting the mean and
    dividing by the variance.
    """

    def __init__(self, do_normaliazion=False, mean=None, var=None):
        self.to_tensor = tf.transforms.ToTensor()
        self.do_norm = do_normaliazion
        self.mean = mean
        self.var = var

    def normalize(self, sample):
        """
        Normalizes the image to have mean = 0 und variance = 1
        :param sample: torch tensor of shape [batch_size, channels, height, width]
        """
        return torch.sub(sample, self.mean) / self.var

    def __call__(self, sample):
        sample = self.to_tensor(sample).double()
        if self.do_norm:
            sample = self.normalize(sample)
        return sample


class ToTorchLong(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return torch.LongTensor(sample)


class PILResizeByFactor(object):
    """Wrapper for the PIL resize method to be able to use it in the Compose structure of the datasets"""

    def __init__(self, interpolation, factor=1):
        self.interpolation = interpolation
        self.f = factor

    def __call__(self, sample):
        return sample.resize([int(sample.size[0]/self.f), int(sample.size[1]/self.f)], resample=self.interpolation)


class PILResize(object):
    """Wrapper for the PIL resize method to be able to use it in the Compose structure of the datasets"""

    def __init__(self, size, interpolation, box=None, reducing_gap=None):
        self.size = size
        self.interpolation = interpolation
        self.box = box
        self.reducing_gap = reducing_gap

    def __call__(self, sample):
        return sample.resize(self.size, resample=self.interpolation, box=self.box, reducing_gap=self.reducing_gap)


class CV2ResizeByFactor(object):
    """Wrapper for the cv2 resize method to be able to use it in the Compose structure of the datasets"""

    def __init__(self,interpolation, factor=1):
        self.interpolation = interpolation
        self.f = factor

    def __call__(self, sample):
        return cv2.resize(sample,
                          dsize=[int(sample.shape[1]/self.f), int(sample.shape[0]/self.f)],
                          interpolation=self.interpolation)


class CV2Resize(object):
    """Wrapper for the cv2 resize method to be able to use it in the Compose structure of the datasets"""

    def __init__(self, size, interpolation, box=None, reducing_gap=None):
        self.size = size
        self.interpolation = interpolation
        self.box = box
        self.reducing_gap = reducing_gap

    def __call__(self, sample):
        return cv2.resize(sample, dsize=self.size, interpolation=self.interpolation)


class PILHorizontalFlip(object):
    """Wrapper for the PIL transpose method to be able to use it in the Compose structure of the datasets"""

    def __init__(self, do_flip):
        self.do_flip = do_flip

    def __call__(self, sample):
        if self.do_flip:
            sample = sample.transpose(pil.FLIP_LEFT_RIGHT)
        else:
            sample = sample
        return sample


class CV2HorizontalFlip(object):
    """Wrapper for the CV2 flip method to be able to use it in the Compose structure of the datasets"""

    def __init__(self, do_flip):
        self.do_flip = do_flip

    def __call__(self, sample):
        if self.do_flip:
            return cv2.flip(sample, 1)  # 1 = flip around y-axis
        else:
            return sample


class ColorAug(object):
    """Wrapper for the PIL transpose method to be able to use it in the Compose structure of the datasets"""

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

# --------------------------------------------Cityscapes----------------------------------
class ToInt32Array(object):
    """Transforms the data (PIL Image) into a numpy int32 array"""

    def __init__(self):
        pass

    def __call__(self, sample):
        return np.array(sample).astype('int32')


class CityscapesEncodeSegmentation(object):
    """
    Encodes valid classes the semantic label of cityscapes into labels of the training ids and the rest to the
    ignore_index
    """

    def __init__(self, valid_classes, class_map, ignore_index, label_color):
        self.valid_classes = valid_classes
        self.class_map = class_map
        self.ignore_index = ignore_index
        self.label_color = label_color

    def encode_segmap(self, lbl):
        lbl_copy = np.zeros(shape=(lbl.shape[0], lbl.shape[1])) + self.ignore_index
        for id, _i in enumerate(self.valid_classes):
            mask = np.all(lbl == self.label_color[id], axis=-1)
            lbl_copy[mask] = self.class_map[_i]
        return lbl_copy

    def __call__(self, sample):
        sample = self.encode_segmap(sample)
        return torch.LongTensor(sample)


class CropCarAway(object):
    """
    Crops the lower 15 percent of the Image away, this is used only for training on the cityscapes sequence dataset.
    The reflectiveness of the engine hood, which is always positioned in the lower half of the image,
    can create unwanted artefacts during training.
    """

    def __init__(self, img_height, img_width):
        self.height, self.width = int(img_height * 0.85), int(img_width * 0.85)

    def __call__(self, sample):
        return torchvision.transforms.functional.crop(img=sample,
                                                      top=0,
                                                      left=0,
                                                      height=self.height,
                                                      width=self.width)


# -------------------------------Synthia Rand Cityscapes----------------------------------
class SynthiaToCityscapesEncoding(object):
    """Encodes the Synthia labels into Cityscapes training labels."""

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
        label_copy = torch.LongTensor(numpy.array(label_copy))
        return label_copy


class TransformToDepthSynthia(object):
    """Transforms the depth ground truth of Synthia given in centimeters into meters."""

    def __init__(self):
        pass

    def __call__(self, sample):
        return np.array(sample).astype(np.float32) / 100


class MaskPixelOutsideDepthRange(object):
    """
    Set the depth value of pixels, that have depth value outside the specified range, to ignore_value.
    """

    def __init__(self, min_depth, max_depth, ignore_value):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.ignore_value = ignore_value

    def __call__(self, sample):
        mask_min = sample < self.min_depth
        mask_max = sample > self.max_depth
        mask_min_max = torch.logical_or(mask_min, mask_max)
        sample[mask_min_max] = self.ignore_value
        return sample


class ToInt64Array(object):
    """Transforms the data (e.g. PIL Image) into an numpy int64 array"""

    def __init__(self):
        pass

    def __call__(self, sample):
        sample = np.asarray(sample, dtype=np.int64)
        return sample

# ------------------------------------------GTA5------------------------------------------
class EncodeSegmentationGTA5(object):
    """Encodes the GTA5 semantic labels into Cityscapes traingin labels"""

    @deprecated
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
        sample = self.encode_segmap(sample)
        return self.encode_segmap(sample)


# ------------------------------------------Kitti-----------------------------------------
class TransformToDepthKitti(object):
    @deprecated
    def __init__(self):
        pass

    def __call__(self, sample):
        return np.array(sample).astype(np.float32) / 256


# ------------------------------------------Kalimu----------------------------------------
class TransformToDepthKalimu(object):
    @deprecated
    def __init__(self):
        pass

    def __call__(self, sample):
        return np.array(sample).astype(np.float32) / 256

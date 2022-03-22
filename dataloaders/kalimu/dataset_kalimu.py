# Own files
from cfg.config_dataset import get_cfg_dataset_defaults
from dataloaders import dataset_base
from dataloaders.dataset_base import PathsHandlerRGB
from misc import transforms as tf_prep

# External libraries
# I/O
import random
import numbers
import os
import os.path
import glob
import ntpath
import torch
import numpy as np
import PIL.Image as pil
from torch.utils.data import DataLoader
from torchvision import transforms


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def decompose_rgb_path(path_rgb_file):
    path_components = {}
    par_dir = path_rgb_file

    # Go through the rgb path and decompose its parent directory names step for step and store in path_components
    for i in range(5):
        par_dir = os.path.dirname(par_dir)
        path_components[i] = os.path.basename(par_dir)

    # Take the path components out as needed
    cam_id = path_components[1]
    drive_seq = path_components[3]
    mode = path_components[4]
    frame_id = os.path.splitext(path_leaf(path_rgb_file))[0]
    format = os.path.splitext(path_leaf(path_rgb_file))[1]

    return mode, drive_seq, cam_id, frame_id, format


# Attention: The user must make sure, that the data is sorted as required by the program. However, consistency is
# guaranteed here. If the kalimu dataset is sorted by the scripts as in this framework, then the correct order is
# also guaranteed.
class _PathsKalimu(PathsHandlerRGB):
    def __init__(self, mode, split, cfg):
        super(_PathsKalimu, self).__init__(mode, split, cfg)

    def get_rgb_image_paths(self, mode, drive_seqs='*', cam_ids='image_blackfly', frame_ids='*', format=".png", split=None):
        if split is None: # i.e. no specific subset of files is given, hence use all data
            a = self.get_rgb_image_path(self.path_base, mode, drive_seqs, cam_ids, frame_ids, format)
            print(a)
            return sorted(
                glob.glob(
                    self.get_rgb_image_path(self.path_base, mode, drive_seqs, cam_ids, frame_ids, format)
                )
            )
        # use a subset of files given in a filter file
        else:
            filter_file = '{}/{}/{}'.format(self.path_base, "splits", split+"_"+mode+".txt")
            with open(filter_file, 'r') as f:
                return sorted(
                    [self.get_rgb_image_path(self.path_base, mode, s.split()[0], s.split()[1][-1], s.split()[2], s.split()[3]) for s in f.read().splitlines()]
                )

    @staticmethod
    def get_rgb_image_path(path_base, mode, drive_seqs, cam_ids, frame_ids, format):
        return '{}/{}/{}/rgb/{}/data/{}{}'.format(path_base, mode, drive_seqs, cam_ids, frame_ids, format)

    @staticmethod
    def get_calib_path(path_base):
        return '{}/calib/calib.txt'.format(path_base)


class KalimuSequenceDataset(dataset_base.DatasetRGB):
    """
    Dataset for the semantically annotated images of the Cityscapes Dataset.
    """

    def __init__(self, mode, _, cfg):
        """
        Based on https://github.com/RogerZhangzz/CAG_UDA/blob/master/data/cityscapes_dataset.py
        Initializes the Cityscapes Semantic dataset by collecting all the paths and random shuffling the data if wanted.
        """
        self.paths = _PathsKalimu(mode, None, cfg)

        super(KalimuSequenceDataset, self).__init__(self.paths, cfg)

        self.cfg = cfg

        self.img_size = cfg.dataset.feed_img_size
        self.img_norm = cfg.dataset.img_norm

        self.ids = np.asarray([i for i in range(len(self.paths.paths_rgb))])

        self.mean = torch.tensor([[[0.210395798, 0.199310437, 0.183492795]]]).transpose(0, 2)
        self.var = torch.tensor([[[1.0, 1.0, 1.0]]]).transpose(0, 2)

        self.do_normalization = self.cfg.dataset.img_norm

        self.aug_params = {"brightness_jitter": cfg.dataset.augmentation.brightness_jitter,
                           "contrast_jitter": cfg.dataset.augmentation.contrast_jitter,
                           "saturation_jitter": cfg.dataset.augmentation.saturation_jitter,
                           "hue_jitter": cfg.dataset.augmentation.hue_jitter}

    def __len__(self):
        """
        :return: The number of ids in the split of the dataset.
        """
        return len(self.ids)

    def __getitem__(self, index):
        """
        Collects the rgb images of the sequence and the label of the index image and returns them as a dict.
        Images can be accessed by dict_variable[("rgb", offset)] with e.g. offset = 0 for the index image.
        Labels can be accessed by dict_variable["gt"].
        :param index: Index of an image in the sequence.
        :return: dict of images and label
        """
        # get the shuffled id, index and id are the same if cfg.dataset.shuffle set to false
        index = self.ids[index]
        rgb_imgs = {}
        for offset in self.rgb_frame_offsets:
            if self.get_rgb(self.paths.paths_rgb[index], offset) is not None:
                rgb_imgs[offset] = self.get_rgb(self.paths.paths_rgb[index], offset)
            else:  # As we can't train on the first and last image as temporal information is missing, we append the
                # dataset by two images for the beginning and the end by the corresponding non-offset RGB image
                rgb_imgs[offset] = self.get_rgb(self.paths.paths_rgb[index], 0)

        if self.mode == 'train':
            rgb_imgs = self.transform_train(rgb_imgs)
        elif self.mode == 'val' or self.mode =='test':
            rgb_imgs = self.transform_val(rgb_imgs)
        else:
            assert False, "The mode {} is not defined!".format(self.mode)

        # Create the map to be outputted
        data = {}

        for offset, val in rgb_imgs.items():
            data[("rgb", offset)] = val
        return data

    def transform_train(self, rgb_dict):

        """
        Transforms the rgb images and the semantic ground truth for training.
        :param rgb_dict: dict of rgb images of a sequence.
        :param gt_semantic: ground truth of the image with offset = 0
        :return: dict of transformed rgb images and transformed label
        """
        do_flip = False  # random.random() > 0.5
        do_aug = False  # random.random() > 0.5

        # Get the transformation objects
        tf_rgb_train = self.tf_rgb_train(tgt_size=self.feed_img_size,
                                         do_flip=do_flip,
                                         do_aug=do_aug,
                                         aug_params=self.aug_params,
                                         do_normalization=self.do_normalization,
                                         mean=self.mean,
                                         var=self.var)

        rgb_dict_tf = {}
        for k, img in rgb_dict.items():
            rgb_dict_tf[k] = tf_rgb_train(img) if img is not None else None

        return rgb_dict_tf

    def transform_val(self, rgb_dict):
        """
        Transforms the rgb images and the semantic ground truth for validation.
        :param rgb_dict: dict of rgb images of a sequence.
        :param gt_semantic: ground truth of the image with offset = 0
        :return: dict of transformed rgb images and transformed label
        """
        # Get the transformation objects
        tf_rgb_val = self.tf_rgb_val(tgt_size=self.feed_img_size,
                                     do_normalization=self.do_normalization,
                                     mean=self.mean,
                                     var=self.var)

        rgb_dict_tf = {}
        for k, img in rgb_dict.items():
            rgb_dict_tf[k] = tf_rgb_val(img) if img is not None else None

        return rgb_dict_tf

    def get_rgb(self, path_file, offset):
        assert isinstance(offset, numbers.Number), "The inputted offset {} is not numeric!".format(offset)
        assert os.path.exists(path_file), "The file {} does not exist!".format(path_file)

        tgt_path_file = path_file

        if offset != 0:
            mode, drive_seq, cam_id, frame_id, format_img = decompose_rgb_path(path_file)
            tgt_frame_id = "{:010d}".format(int(frame_id) + offset)
            tgt_path_file = self.paths.get_rgb_image_path(self.paths.path_base, mode, drive_seq, cam_id, tgt_frame_id, format_img)

        if not os.path.exists(tgt_path_file):
            return None

        with open(tgt_path_file, 'rb') as f:
            print(tgt_path_file)
            return pil.open(f).convert('RGB')

    @staticmethod
    def tf_rgb_train(tgt_size, do_flip, do_aug, aug_params, do_normalization, mean, var):
        """
        Transformations of the rgb image during training.
        :param mean: mean of rgb images
        :param tgt_size: target size of the images after resize operation
        :param do_flip: True if the image should be horizontally flipped else False
        :param do_aug: True if augmentations should be applied to the rgb image else False
        :param aug_params: augmentation parameters
        :param do_normalization: true if image should be normalized
        :param mean: mean of R,G,B of rgb images
        :param var: variance in R,G,B of rgb images
        :return: Transformation composition
        """
        return transforms.Compose(
            [
                tf_prep.PILResize(tgt_size, pil.LANCZOS),
                tf_prep.PILHorizontalFlip(do_flip),
                tf_prep.ColorAug(do_aug, aug_params),
                tf_prep.PrepareForNet(do_normalization, mean, var)
            ]
        )

    @staticmethod
    def tf_rgb_val(tgt_size, do_normalization, mean, var):
        """
        Transformations of the rgb image during validation.
        :param mean: mean of rgb images
        :param tgt_size: target size of the images after resize operation
        :param do_normalization: true if image should be normalized
        :param mean: mean of R,G,B of rgb images
        :param var: variance in R,G,B of rgb images
        :return: Transformation composition
        """
        return transforms.Compose(
            [
                tf_prep.PILResize(tgt_size, pil.LANCZOS),
                tf_prep.PrepareForNet(do_normalization, mean, var)
            ]
        )


if __name__ == '__main__':
    path = r'C:\Users\benba\Documents\University\Masterarbeit\Depth-Semantic-UDA\cfg\yaml_files\train\kalimu_guda\kalimu.yaml'
    cfg = get_cfg_dataset_defaults()
    cfg.merge_from_file(path)
    cfg.freeze()

    kalimu = KalimuSequenceDataset('val', None, cfg)
    loader = DataLoader(kalimu,
                        batch_size=1,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False)

    import matplotlib.pyplot as plt

    for idx, data in enumerate(loader):
        fig, axs = plt.subplots(1, 3)
        for i, offset in enumerate([-4, 0, +4]):
            axs[i].imshow(data[("rgb", offset)][0].numpy().transpose(1, 2, 0))
        plt.show()

# Own files
import torch
import wandb
from torch.utils.data import DataLoader

from misc import transforms as tf_prep
from dataloaders import dataset_base

# External libraries
# I/O
import os
import os.path
import glob
import scipy.io as io
import numpy as np
import re
import ntpath
# Image processing
import PIL.Image as pil
from torchvision import transforms
import matplotlib.pyplot as plt

# Miscellaneous
import numbers
import random


def decompose_rgb_path(path_file):
    """
    Decomposes the path to the rgb file and splits the name of the file into meaningful components
    :param path_file: path to the file.
    :return:
        base_path: base path to the dataset
        mode: 'train', 'val', 'test'
        city: city where the data has been collected
        seq_number: number of the sequence given a specific city
        frame_id: number of the frame within the specific sequence
        data_type: defines which camera has been used and if the data is raw data or ground truth, e.g. 'leftImg8bit'
        file_format: extension of the file e.g. '.png'
    """
    path_components = {}
    par_dir = path_file

    # Go through the rgb path and decompose its parent directory names step for step and store in path_components
    for i in range(3):
        path_components[i] = os.path.basename(par_dir)
        par_dir = os.path.dirname(par_dir)

    # Take the path components out as needed

    city, seq_number, frame_id, data_type = re.split('_', path_components[0])
    file_format = re.search('(\.[A-Za-z0-9]*)', data_type).group(0)
    data_type = data_type[:-len(file_format)]  # cut off extension to get data type e.g 'leftImg8bit'
    mode = path_components[2]  # eg. 'leftImg8bit'

    return par_dir, mode, city, seq_number, frame_id, data_type, file_format


class _PathsCityscapesSequence(dataset_base.PathsHandlerRGB):
    def __init__(self, mode, cfg):
        """
        :param mode: the mode ('train', 'val', 'test') in which the data is used.
        :param cfg: the configuration.
        """
        super(_PathsCityscapesSequence, self).__init__(mode, None, cfg)

    def get_rgb_image_paths(self, mode, cities='*', frame_ids='*', seq_ids='*', data_type='*', file_format=".png",
                            **kwargs):
        """
        Fetch all paths to the rgb images.
        :param data_type: type of the dataset used e.g. 'leftImg8bit' or 'leftImg8bit_sequence' or the default '*'
        :param mode: 'train', 'test', 'val' possible values
        :param cities: '*', or any city in the dataset e.g. 'aachen'
        :param frame_ids: '*' or specific 6 digit string
        :param seq_ids: '*' or specific 6 digit string
        :param file_format: the file extension e.g '.png'
        :return: List of paths to all exsting files in the specific subset (mode)
        """
        return sorted(
            glob.glob(
                self.get_rgb_image_path(self.path_base, mode, cities, seq_ids, frame_ids, data_type, file_format)
            )
        )

    @staticmethod
    def get_rgb_image_path(path_base, mode, cities, seq_ids, frame_ids, data_type, file_format):
        """
        default path: base_path/images/<mode>/*/*_*_*.png
        example specific path: base_path/images/train/aachen/aachen_000000_000019_leftImg8bit.png
        :param data_type: type of the dataset used e.g. 'leftImg8bit' or 'leftImg8bit_sequence' or the default '*'
        :param mode: 'train', 'test', 'val' possible values
        :param seq_ids: '*' or specific 6 digit string
        :param cities: '*', or any city in the dataset e.g. 'aachen'
        :param path_base: path to the base dataset e.g '<path_to_project>/data/Cityscapes_datasets/leftImg8bit'
        :param frame_ids: '*' or specific 6 digit string
        :param file_format: the file extension e.g '.png'
        :return: single specific path if files exists else None
        """
        path = os.path.join(path_base, mode, cities, cities + '_' + seq_ids + '_' + frame_ids + '_' + data_type +
                            file_format)
        if os.path.exists(path) or \
                not frame_ids.lstrip("-").isnumeric() or \
                not seq_ids.lstrip("-").isnumeric() or \
                data_type == '*' or \
                cities == '*':
            # exact path exists or it is a path including '*'
            # isnumeric returns false on negative numbers '-001'
            return path
        else:
            # path is a specific path, but it does not exist on the drive
            return None


class CityscapesSequenceDataset(dataset_base.DatasetRGB):
    """
    Dataset for the semantically annotated images of the Cityscapes Dataset.
    """

    def __init__(self, mode, _, cfg):
        """
        Based on https://github.com/RogerZhangzz/CAG_UDA/blob/master/data/cityscapes_dataset.py
        Initializes the Cityscapes Semantic dataset by collecting all the paths and random shuffling the data if wanted.
        """
        self.paths = _PathsCityscapesSequence(mode, cfg)

        super(CityscapesSequenceDataset, self).__init__(self.paths, cfg)

        self.cfg = cfg

        self.img_size = cfg.dataset.feed_img_size
        self.img_norm = cfg.dataset.img_norm

        self.ids = np.asarray([i for i in range(len(self.paths.paths_rgb))])

        self.mean = torch.tensor([[[0.28689554, 0.32513303, 0.28389177]]]).transpose(0, 2)
        self.var = torch.tensor([[[0.18696375, 0.19017339, 0.18720214]]]).transpose(0, 2)

        self.do_normalization = self.cfg.dataset.img_norm

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
        do_flip = random.random() > 0.5

        # Get the transformation objects
        tf_rgb_train = self.tf_rgb_train(tgt_size=self.feed_img_size,
                                         do_flip=do_flip,
                                         do_normalization = self.do_normalization,
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
        """
        Loads the PIL Image with the given offset to the path file label.
        :param path_file: path to the label
        :param offset: offsets of the other rgb images in the sequence to the path_file image.
        :return: dict of Pil RGb Images.
        """
        assert isinstance(offset, numbers.Number), "The inputted offset {} is not numeric!".format(offset)
        assert os.path.exists(path_file), "The file {} does not exist!".format(path_file)

        tgt_path_file = path_file

        parent_dir, mode, city, seq_number, frame_id, data_type, file_format = decompose_rgb_path(path_file)

        if offset != 0:
            tgt_frame_id = '{0:06d}'.format(int(frame_id) + offset)

            tgt_path_file = self.paths.get_rgb_image_path(path_base=parent_dir,
                                                          mode=mode, cities=city,
                                                          frame_ids=tgt_frame_id,
                                                          seq_ids=seq_number,
                                                          data_type=data_type,
                                                          file_format=file_format)

        if tgt_path_file is None:
            return None
        print(tgt_path_file)
        img = pil.open(tgt_path_file)
        return img

    @staticmethod
    def tf_rgb_train(tgt_size, do_flip, do_normalization, mean, var): # fixme add augmentations to train and val rgb transforms
        """
        Transformations of the rgb image during training.
        :param mean: mean of rgb images
        :param tgt_size: target size of the images after resize operation
        :param do_flip: True if the image should be horizontally flipped else False
        :param do_normalization: true if image should be normalized
        :param mean: mean of R,G,B of rgb images
        :param var: variance in R,G,B of rgb images
        :return: Transformation composition
        """
        return transforms.Compose(
            [
                tf_prep.Crop_Car_Away(img_height=1024, img_width=2048),
                tf_prep.PILResize(tgt_size, pil.BILINEAR),
                tf_prep.PILHorizontalFlip(do_flip),
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
                tf_prep.Crop_Car_Away(img_height=1024, img_width=2048),
                tf_prep.PILResize(tgt_size, pil.BILINEAR),
                tf_prep.PrepareForNet(do_normalization, mean, var)
            ]
        )


if __name__ == "__main__":
    from cfg.config_dataset import get_cfg_dataset_defaults
    import sys

    path = sys.argv[1]
    cfg = get_cfg_dataset_defaults()
    cfg.merge_from_file(path)
    cfg.freeze()

    CITY_dataset = CityscapesSequenceDataset("train", None, cfg)

    wandb.init(project='dataset-cityscapes-sequence')

    ds = DataLoader(CITY_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    wandb.log({'len_dataset': len(ds)})

    for i, data in enumerate(ds):
        img0 = wandb.Image(data[('rgb', 0)].squeeze(0).numpy().transpose(1, 2, 0), caption="RGB 0")
        img_minus_1 = wandb.Image(data[('rgb', -1)].squeeze(0).numpy().transpose(1, 2, 0), caption="RGB -1")
        img_plus_1 = wandb.Image(data[('rgb', 1)].squeeze(0).numpy().transpose(1, 2, 0), caption="RGB +1")
        wandb.log({'images': [img0, img_minus_1, img_plus_1]})
        if i == 10:
            break

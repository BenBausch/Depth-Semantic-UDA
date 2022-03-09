# Own files
import warnings

import torch
from torch.utils.data import DataLoader
from misc import transforms as tf_prep
from dataloaders import dataset_base
from utils.constans import IGNORE_INDEX_SEMANTIC

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


class _PathsCityscapesSemantic(dataset_base.PathsHandlerSemantic):
    def __init__(self, mode, cfg):
        """
        :param mode: the mode ('train', 'val', 'test') in which the data is used.
        :param cfg: the configuration.
        """
        super(_PathsCityscapesSemantic, self).__init__(mode, None, cfg)

    def get_rgb_image_paths(self, mode, cities='*', frame_ids='*', seq_ids='*', data_type='*', file_format=".png",
                            **kwargs):
        """
        Fetch all paths to the images of split <mode>.
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

    def get_semantic_label_paths(self, mode, cities='*', frame_ids='*', seq_ids='*', data_type='*', file_format=".png",
                                 **kwargs):
        """
        Fetch all paths to the labels of split <mode>.
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
                self.get_semantic_label_path(self.path_base, mode, cities, seq_ids, frame_ids, file_format)
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
        path = os.path.join(path_base, 'images', mode, cities, cities + '_' + seq_ids + '_' + frame_ids + '_' +
                            data_type + file_format)
        if os.path.exists(path) or \
                not frame_ids.isnumeric() or \
                not seq_ids.isnumeric() or \
                data_type == '*' or \
                cities == '*':
            # exact path exists or it is a path including '*'
            return path
        else:
            # path is a specific path, but it does not exist on the drive
            return None

    @staticmethod
    def get_semantic_label_path(path_base, mode, cities, seq_ids, frame_ids, file_format):
        """
        default path: base_path/labels/<mode>/*/*_*_*.png
        example specific path: base_path/labels/train/aachen/aachen_000000_000019_leftImg8bit.png
        :param mode: 'train', 'test', 'val' possible values
        :param seq_ids: '*' or specific 6 digit string
        :param cities: '*', or any city in the dataset e.g. 'aachen'
        :param path_base: path to the base dataset e.g '<path_to_project>/data/Cityscapes_datasets/leftImg8bit'
        :param frame_ids: '*' or specific 6 digit string
        :param file_format: the file extension e.g '.png'
        :return: single specific path if files exists else None
        """
        # Only use the semantic labels ( other data is for traffic participating individuals like a specific person)
        data_type = 'gtFine_color'

        path = os.path.join(path_base, 'labels', mode, cities, cities + '_' + seq_ids + '_' + frame_ids + '_' +
                            data_type + file_format)
        if os.path.exists(path) or \
                not frame_ids.isnumeric() or \
                not seq_ids.isnumeric() or \
                data_type == '*' or \
                cities == '*':
            # exact path exists or it is a path including '*'
            return path
        else:
            # path is a specific path, but it does not exist on the drive
            return None


class CityscapesSemanticDataset(dataset_base.DatasetRGB, dataset_base.DatasetSemantic):
    """
    Dataset for the semantically annotated images of the Cityscapes Dataset.
    """

    def __init__(self, mode, _, cfg):
        """
        Based on https://github.com/RogerZhangzz/CAG_UDA/blob/master/data/cityscapes_dataset.py
        Initializes the Cityscapes Semantic dataset by collecting all the paths and random shuffling the data if wanted.
        """

        self.paths = _PathsCityscapesSemantic(mode, cfg)

        # The dataset does not contain sequences therefore the offsets refer only to the selected RGB image(offset == 0)
        assert len(cfg.dataset.rgb_frame_offsets) == 1
        assert cfg.dataset.rgb_frame_offsets[0] == 0

        super(CityscapesSemanticDataset, self).__init__(self.paths, cfg)

        self.cfg = cfg

        self.n_classes = self.cfg.dataset.num_classes

        if self.n_classes == 19:
            self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]  # in cityscapes ids
            self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32,
                                  33]  # in cityscapes ids
            self.colors = [[128, 64, 128],  # road
                           [244, 35, 232],  # sidewalk
                           [70, 70, 70],  # building
                           [102, 102, 156],  # wall
                           [190, 153, 153],  # fence
                           [153, 153, 153],  # pole
                           [250, 170, 30],  # traffic_light
                           [220, 220, 0],  # traffic_sign
                           [107, 142, 35],  # vegetation
                           [152, 251, 152],  # terrain
                           [70, 130, 180],  # sky
                           [220, 20, 60],  # person
                           [255, 0, 0],  # rider
                           [0, 0, 142],  # car
                           [0, 0, 70],  # truck
                           [0, 60, 100],  # bus
                           [0, 80, 100],  # train
                           [0, 0, 230],  # motorcycle
                           [119, 11, 32]]  # bicycle
            self.class_names = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
                                "traffic_sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck",
                                "bus", "train", "motorcycle", "bicycle"]  # only the valid class names
        elif self.n_classes == 16:
            self.colors = [[128, 64, 128],  # road
                           [244, 35, 232],  # sidewalk
                           [70, 70, 70],  # building
                           [102, 102, 156],  # wall
                           [190, 153, 153],  # fence
                           [153, 153, 153],  # pole
                           [250, 170, 30],  # traffic_light
                           [220, 220, 0],  # traffic_sign
                           [107, 142, 35],  # vegetation
                           [70, 130, 180],  # sky
                           [220, 20, 60],  # person
                           [255, 0, 0],  # rider
                           [0, 0, 142],  # car
                           [0, 60, 100],  # bus
                           [0, 0, 230],  # motorcycle
                           [119, 11, 32]]  # bicycle
            self.class_names = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
                                "traffic_sign", "vegetation", "sky", "person", "rider", "car",
                                "bus", "motorcycle", "bicycle"]  # only the valid class names
            self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 22, 27, 29, 30, 31,
                                 -1]  # in cityscapes ids
            self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 23, 24, 25, 26, 28, 32, 33]  # in cityscapes ids
        else:
            raise ValueError('Cityscapes is not implemented for {self.n_classes} classes!')

        self.label_colours = dict(zip(range(self.n_classes), self.colors))
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))

        if cfg.dataset.debug:
            print('Training on the following classes:\n name | cityscapes_id | training_id ')
            for i in self.valid_classes:
                print(f'{self.class_names[self.class_map[i]]:<15} | {str(i):<2} | {self.class_map[i]:<2}')

        self.ignore_index = IGNORE_INDEX_SEMANTIC

        self.img_size = cfg.dataset.feed_img_size
        self.img_norm = cfg.dataset.img_norm

        self.mean = torch.tensor([[[0.28689554, 0.32513303, 0.28389177]]]).transpose(0, 2)
        self.var = torch.tensor([[[1.0, 1.0, 1.0]]]).transpose(0, 2)

        self.do_normalization = self.cfg.dataset.img_norm

        self.ids = np.asarray([i for i in range(len(self.paths.paths_rgb))])

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

        # Get all required training elements
        gt_semantic = self.get_semantic(
            self.paths.paths_semantic[index]) if self.paths.paths_semantic is not None else None

        rgb_imgs = {}
        for offset in self.rgb_frame_offsets:
            if self.get_rgb(self.paths.paths_rgb[index], offset) is not None:
                rgb_imgs[offset] = self.get_rgb(self.paths.paths_rgb[index], offset)
            else:  # As we can't train on the first and last image as temporal information is missing, we append the
                # dataset by two images for the beginning and the end by the corresponding non-offset RGB image
                rgb_imgs[offset] = self.get_rgb(self.paths.paths_rgb[index], 0)

        if self.mode == 'train':
            rgb_imgs, gt_semantic = self.transform_train(rgb_imgs, gt_semantic)
        elif self.mode == 'val' or self.mode == 'test':
            rgb_imgs, gt_semantic = self.transform_val(rgb_imgs, gt_semantic)
        else:
            assert False, "The mode {} is not defined!".format(self.mode)

        # Create the map to be outputted
        data = {}

        if gt_semantic is not None:
            data["semantic"] = gt_semantic

        for offset, val in rgb_imgs.items():
            data[("rgb", offset)] = val

        return data

    def transform_train(self, rgb_dict, gt_semantic):
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
                                         do_normalization=self.do_normalization,
                                         mean=self.mean,
                                         var=self.var)

        tf_semantic_train = self.tf_semantic_train(self.feed_img_size,
                                                   do_flip,
                                                   self.valid_classes,
                                                   self.class_map,
                                                   self.ignore_index,
                                                   self.label_colours)

        rgb_dict_tf = {}
        for k, img in rgb_dict.items():
            rgb_dict_tf[k] = tf_rgb_train(img) if img is not None else None

        gt_semantic = tf_semantic_train(gt_semantic) if gt_semantic is not None else None

        return rgb_dict_tf, gt_semantic

    def transform_val(self, rgb_dict, gt_semantic):
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
        tf_semantic_val = self.tf_semantic_val(self.feed_img_size,
                                               self.valid_classes,
                                               self.class_map,
                                               self.ignore_index,
                                               self.label_colours)

        rgb_dict_tf = {}
        for k, img in rgb_dict.items():
            rgb_dict_tf[k] = tf_rgb_val(img) if img is not None else None

        gt_semantic = tf_semantic_val(gt_semantic) if gt_semantic is not None else None

        return rgb_dict_tf, gt_semantic

    def get_semantic(self, path_file):
        """
        Loads the PIL Image for the path_file label if it exists.
        :param path_file: path to the label
        :return: Pil Image of semantic ground truth.
        """
        if path_file is None:
            return None
        else:
            assert os.path.exists(path_file), "The file {} does not exist!".format(path_file)
            label = pil.open(path_file).convert('RGB')
            return label

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

            tgt_path_file = self.paths.get_rgb_image_path(mode, cities=city, frame_ids=tgt_frame_id, seq_ids=seq_number,
                                                          data_type=data_type, file_format=file_format)

        if tgt_path_file is None:
            return None
        img = pil.open(tgt_path_file).convert('RGB')
        return img

    @staticmethod
    def tf_rgb_train(tgt_size, do_flip, do_normalization, mean,
                     var):  # fixme add augmentations to train and val rgb transforms
        """
        Transformations of the rgb image during training.
        :param tgt_size: target size of the images after resize operation
        :param do_flip: True if the image should be horizontally flipped else False
        :param do_normalization: true if image should be normalized
        :param mean: mean of R,G,B of rgb images
        :param var: variance in R,G,B of rgb images
        :return: Transformation composition
        """
        return transforms.Compose(
            [
                tf_prep.PILResize(tgt_size, pil.BILINEAR),
                tf_prep.PILHorizontalFlip(do_flip),
                tf_prep.PrepareForNet(do_normalization, mean, var)
            ]
        )

    @staticmethod
    def tf_semantic_train(tgt_size, do_flip, valid_classes, class_map, ignore_index, label_colors):
        """
        Transformations of the label during training.
        :param label_colors: dict mapping valid class id to its rgb color
        :param ignore_index: pixel will be labeled ignore_index if they are not part of a valid class
        :param class_map: dict mapping valid class ids to numbers
        :param void_classes: list of non valid classes (such pixel will be set to ignore index)
        :param tgt_size: target size of the labels after resize operation
        :param do_flip: True if the image should be horizontally flipped else False
        :return: Transformation composition
        """
        return transforms.Compose(
            [
                tf_prep.PILResize(tgt_size, pil.NEAREST),
                tf_prep.PILHorizontalFlip(do_flip),
                tf_prep.ToInt32Array(),
                tf_prep.CityscapesEncodeSegmentation(valid_classes, class_map, ignore_index, label_colors)
            ]
        )

    @staticmethod
    def tf_rgb_val(tgt_size, do_normalization, mean, var):
        """
        Transformations of the rgb image during validation.
        :param tgt_size: target size of the images after resize operation
        :param do_normalization: true if image should be normalized
        :param mean: mean of R,G,B of rgb images
        :param var: variance in R,G,B of rgb images
        :return: Transformation composition
        """
        return transforms.Compose(
            [
                tf_prep.PILResize(tgt_size, pil.BILINEAR),
                tf_prep.PrepareForNet(do_normalization, mean, var)
            ]
        )

    @staticmethod
    def tf_semantic_val(tgt_size, valid_classes, class_map, ignore_index, label_colors):
        """
        Transformations of the labels during validation.
        :param label_colors: dict mapping valid class id to its rgb color
        :param ignore_index: pixel will be labeled ignore_index if they are not part of a valid class
        :param class_map: dict mapping valid class ids to numbers
        :param valid_classes: list of valid classes
        :param tgt_size: target size of the labels after resize operation
        :return: Transformation composition
        """
        return transforms.Compose(
            [
                tf_prep.PILResize(tgt_size, pil.NEAREST),
                tf_prep.ToInt32Array(),
                tf_prep.CityscapesEncodeSegmentation(valid_classes, class_map, ignore_index, label_colors)
            ]
        )

    def get_valid_ids_and_names(self):
        """Returns valid training class ids and the corresponding class names."""
        ids = []
        names = []
        for idx in self.valid_classes:
            city_id = self.class_map[idx]
            ids.append(city_id)
            names.append(self.class_names[city_id])
        return ids, names

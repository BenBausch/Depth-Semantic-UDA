# Project Imports
import torch
from torch import tensor
from torch.utils.data import DataLoader

from dataloaders.dataset_base import DatasetDepth, DatasetSemantic, DatasetRGB, \
    PathsHandlerDepthDense, PathsHandlerSemantic
from dataloaders.scritps.prepare_synthia import available_splits
from misc import transforms as tf_prep

# Packages
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import random
import glob
import os
import PIL.Image as pil
import cv2
from torchvision import transforms
import re

from utils.constans import IGNORE_VALUE_DEPTH, IGNORE_INDEX_SEMANTIC


def decompose_rgb_path(path_file):
    """
    Decomposes the path to instance and splits the path of the file into meaningful components
    :param path_file: path to the file.
    :return:
        base_path: base path to the dataset
        information_type: e.g. 'DEPTH', 'RGB', 'MASK'
        class_name: e.g 'person', 'rider'
        instance_id: id of the instance
        file_format: extension of the file e.g. '.png'
    """
    path_components = {}
    par_dir = path_file

    # Go through the rgb path and decompose its parent directory names step for step and store in path_components
    for i in range(3):
        path_components[i] = os.path.basename(par_dir)
        par_dir = os.path.dirname(par_dir)

    # Take the path components out as needed

    file_format = re.search('(\.[A-Za-z0-9]*)', path_components[0]).group(0)
    instance_id = re.search('([A-Za-z0-9]*)\.', path_components[0]).group(0)
    instance_id = instance_id[:-1]  # remove '.' from id

    # example of return: [basepath, 'DENSE', 'person', '60', '.png']
    return [par_dir, path_components[2], path_components[1], instance_id, file_format]


class _PathsSynthiaInstances(PathsHandlerSemantic, PathsHandlerDepthDense):
    """
    Class for handling the paths to the images and labels.
    """

    def __init__(self, mode, _, cfg):
        """
        :param mode: mode of usage: 'train', 'val', 'test'
        :param cfg: configuration
        """

        super(_PathsSynthiaInstances, self).__init__(mode, None, cfg)

    def get_rgb_image_paths(self, mode, class_name='*', frame_ids='*', file_format=".png", *args, **kwargs):
        """
        Gets all the RGB image paths.
        :param mode: mode of usage: 'train', 'val', 'test'
        :param class_name: name of the class of the instance, e.g 'person'
        :param frame_ids: ids of frames to be selected, * = all images
        :param file_format: format of the files
        :param split: split name of the data
        """
        return sorted(
            glob.glob(
                self.get_rgb_image_path(self.path_base, class_name, frame_ids, file_format)
            )
        )

    def get_semantic_label_paths(self, mode, class_name='*', frame_ids='*', file_format=".png", *args, **kwargs):
        """
        Gets all the semantic label paths.
        :param mode: mode of usage: 'train', 'val', 'test'
        :param class_name: name of the class of the instance, e.g 'person'
        :param frame_ids: ids of frames to be selected, * = all images
        :param file_format: format of the files
        :param split: split name of the data
        """
        return sorted(
            glob.glob(
                self.get_semantic_label_path(self.path_base, class_name, frame_ids, file_format)
            )
        )

    def get_gt_depth_image_paths(self, mode, class_name='*', frame_ids='*', file_format=".png", *args, **kwargs):
        """
        Gets all the depth label paths.
        :param mode: mode of usage: 'train', 'val', 'test'
        :param class_name: name of the class of the instance, e.g 'person'
        :param frame_ids: ids of frames to be selected, * = all images
        :param file_format: format of the files
        :param split: split name of the data
        """
        return sorted(
            glob.glob(
                self.get_depth_label_path(self.path_base, class_name, frame_ids, file_format)
            )
        )

    @staticmethod
    def get_rgb_image_path(path_base, class_name, frame_ids, file_format):
        """
            Builds path to an rgb image.
            Default path: path_base/RGB/*.png
            Specific path: path_base/RGB/0000001.png
            :param path_base: path to directory containing images and labels
            :param frame_ids: 7-digit string id or '*'
            :param file_format: extensions of files '.png'
            :return: single path string
        """
        path = os.path.join(path_base, 'RGB', class_name, frame_ids + file_format)
        if os.path.exists(path) or not frame_ids.isnumeric():
            # Apply path exists check only if exact id is provided, no regex like '*'
            return path
        else:
            return None

    @staticmethod
    def get_semantic_label_path(path_base, class_name, frame_ids, file_format):
        """
            Builds path to an semantic label image.
            Default path: path_base/GT/LABEL/*.png
            Specific path: path_base/GT/LABEL/0000001.png
            :param path_base: path to directory containing images and labels
            :param frame_ids: 7-digit string id or '*'
            :param file_format: extensions of files '.png'
            :return: single path string
        """
        path = os.path.join(path_base, 'MASK', class_name, frame_ids + file_format)
        if os.path.exists(path) or not frame_ids.isnumeric():
            # Apply path exists check only if exact id is provided, no regex like '*'
            return path
        else:
            return None

    @staticmethod
    def get_depth_label_path(path_base, class_name, frame_ids, file_format):
        """
            Builds path to an rgb image.
            Default path: path_base/RGB/*.png
            Specific path: path_base/RGB/0000001.png
            :param path_base: path to directory containing images and labels
            :param frame_ids: 7-digit string id or '*'
            :param file_format: extensions of files '.png'
            :return: single path string
        """
        path = os.path.join(path_base, 'DEPTH', class_name, frame_ids + file_format)
        if os.path.exists(path) or not frame_ids.isnumeric():
            # Apply path exists check only if exact id is provided, no regex like '*'
            return path
        else:
            return None


class SynthiaInstancesDataset(DatasetRGB, DatasetSemantic, DatasetDepth):
    """
        RGB Images annotated with Depth and Semantic classes. This dataset does not provide any image sequences.
    """

    def __init__(self, mode, split, cfg):
        """
        :param mode: mode of usage, synthia only supports 'train'
        :param split: split name of the data, currently only None is supported
        :param cfg: configuration
        """
        # The dataset does not contain sequences therefore the offsets refer only to the selected RGB image(offset == 0)
        assert len(cfg.dataset.rgb_frame_offsets) == 1
        assert cfg.dataset.rgb_frame_offsets[0] == 0
        assert split is None, 'Synthia_Rand_Cityscapes currently does not support the spilt of the data!'
        assert mode == 'train', 'Synthia_Rand_Cityscapes does not support any other mode than train!'

        self.paths = _PathsSynthiaInstances(mode, None, cfg)

        super(SynthiaInstancesDataset, self).__init__(pathsObj=self.paths, cfg=cfg)

        # Stuff related to class encoding -------------------------------------------------------------

        # classes considered void have no equivalent valid label in cityscapes
        # See: https://www.cityscapes-dataset.com/dataset-overview/ for void and valid classes
        if self.cfg.dataset.num_classes != 16:
            raise ValueError(f'Synthia not implemented for {self.cfg.dataset.num_classes} classes. \n'
                             f'Only compatible with Cityscapes with 16 classes: ids from 0 to 15!')

        self.ignore_index = IGNORE_INDEX_SEMANTIC
        self.ignore_value = IGNORE_VALUE_DEPTH

        # Stuff related to augmentations --------------------------------------------------------------

        self.do_normalization = self.cfg.dataset.img_norm

        # todo: validate these values
        self.mean = torch.tensor([[[0.314747602, 0.277402550, 0.248091921]]]).transpose(0, 2)
        self.var = tensor([[[1.0, 1.0, 1.0]]]).transpose(0, 2)

        self.resize_factor = self.cfg.dataset.resize_factor

    def __len__(self):
        """
        Returns the number of rgb images for the mode in the split.
        :return:
        """
        return len(self.paths.paths_rgb)

    def __getitem__(self, index):
        """
            Collects the rgb images of the sequence and the label of the index image and returns them as a dict.
            Images can be accessed by dict_variable[("rgb", offset)] with e.g. offset = 0 for the index image.
            Labels can be accessed by dict_variable["gt"].
            :param index: Index of an image with offset 0 in the sequence.
            :return: dict of images and label
        """
        rgb_path_file = self.paths.paths_rgb[index]
        path_components = decompose_rgb_path(rgb_path_file)  # [basepath, 'DENSE', 'person', '60', '.png']
        base_path = path_components[0]
        class_name = path_components[2]
        instance_id = path_components[3]
        file_format = path_components[4]

        rgb_img = self.get_rgb(rgb_path_file)

        # Get all required training elements
        semantic_path_file = os.path.join(base_path, 'MASK', class_name, instance_id + file_format)
        gt_semantic = self.get_semantic(semantic_path_file)

        depth_path_file = os.path.join(base_path, 'DEPTH', class_name, instance_id + file_format)
        gt_depth_dense = self.get_depth(depth_path_file)

        if self.mode == 'train':
            rgb_img, gt_semantic, gt_depth_dense = self.transform_train(rgb_img, gt_semantic, gt_depth_dense)
        elif self.mode == 'val' or self.mode == 'test':
            raise ValueError("Synthia only available for training")
        else:
            assert False, "The mode {} is not defined!".format(self.mode)

        # Create the map to be outputted
        data = {}

        data["semantic"] = gt_semantic

        data["depth_dense"] = gt_depth_dense

        data['rgb'] = rgb_img

        return data

    def transform_train(self, rgb_img, gt_semantic, gt_depth_dense):
        """
            Transforms the rgb images and the semantic ground truth for training.
            :param gt_depth_dense: depth ground truth of image with offset = 0
            :param rgb_img: rgb image.
            :param gt_semantic: semantic ground truth of the image with offset = 0
            :return: dict of transformed rgb images and transformed label
        """
        if self.cfg.dataset.debug:
            do_flip = random.random() > 0.5
            do_aug = False
        else:
            do_flip = random.random() > 0.5
            do_aug = random.random() > 0.5

        # Get the transformation objects
        tf_rgb_train = self.tf_rgb_train(tgt_size=self.feed_img_size,
                                         do_flip=do_flip,
                                         do_normalization=self.do_normalization,
                                         mean=self.mean,
                                         var=self.var,
                                         resize_factor=self.resize_factor)

        tf_semantic_train = self.tf_semantic_train(tgt_size=self.feed_img_size,
                                                   do_flip=do_flip,
                                                   resize_factor=self.resize_factor)

        tf_depth_dense_train = self.tf_depth_train(tgt_size=self.feed_img_size,
                                                   do_flip=do_flip,
                                                   resize_factor=self.resize_factor)

        # Apply transformations

        rgb_img = tf_rgb_train(rgb_img)

        gt_semantic = tf_semantic_train(gt_semantic) if gt_semantic is not None else None

        gt_depth_dense = tf_depth_dense_train(gt_depth_dense) if gt_depth_dense is not None else None

        return rgb_img, gt_semantic, gt_depth_dense

    def get_rgb(self, path_file, *args):
        """
        Loads the PIL Image form the path_file.

        :param path_file: path to the image
        :return: PIL RGB Images.
        """
        if path_file is None:
            return None
        else:
            assert os.path.exists(path_file), "The file {} does not exist!".format(path_file)
            img = pil.open(path_file).convert('RGB')
            return img

    def get_semantic(self, path_file):
        """
            Loads the Labels with opencv (loading with pillow destroys encoding).
            :param path_file: path to the label
            :return: numpy ndarray of semantic ground truth.
        """
        if path_file is None:
            return None
        else:
            assert os.path.exists(path_file), "The file {} does not exist!".format(path_file)

            label = cv2.imread(path_file, cv2.IMREAD_UNCHANGED)
            return label

    def get_depth(self, path_file):
        """
            Loads the depth as PIL Image form the path_file.
            :param path_file: path to the label
            :return: PIL RGB Images.
        """
        if path_file is None:
            return None
        else:
            assert os.path.exists(path_file), "The file {} does not exist!".format(path_file)

            label = cv2.imread(path_file, cv2.IMREAD_UNCHANGED).astype(np.int32)
            return label

    @staticmethod
    def tf_rgb_train(tgt_size, do_flip, do_normalization, mean, var, resize_factor):
        """
        Transformations of the rgb image during training.
        :param resize_factor: factor by which to down sample the resolution, e.g 4 --> 1/4 resolution
        :param tgt_size: target size of the images after resize operation
        :param do_flip: True if the image should be horizontally flipped else False
        :param do_normalization: true if image should be normalized
        :param mean: mean of R,G,B of rgb images
        :param var: variance in R,G,B of rgb images

        :return: Transformation composition
        """
        return transforms.Compose(
            [
                tf_prep.PILResizeByFactor(pil.BICUBIC, resize_factor),
                tf_prep.PILHorizontalFlip(do_flip),
                tf_prep.PrepareForNet(do_normalization, mean, var)
            ]
        )

    @staticmethod
    def tf_semantic_train(tgt_size, do_flip, resize_factor):
        """
            Transformations of the label during training.
            :param tgt_size: target size of the labels after resize operation
            :param do_flip: True if the image should be horizontally flipped else False
            :param resize_factor: factor by which to down sample the resolution, e.g 4 --> 1/4 resolution

            :return: Transformation composition
        """
        return transforms.Compose(
            [
                tf_prep.CV2ResizeByFactor(interpolation=cv2.INTER_NEAREST, factor=resize_factor),
                tf_prep.CV2HorizontalFlip(do_flip=do_flip),
                tf_prep.ToInt64Array(),
            ]
        )

    @staticmethod
    def tf_depth_train(tgt_size, do_flip, resize_factor):
        """
            Transformations of the depth label during training.
            :param tgt_size: target size of the labels after resize operation
            :param do_flip: True if the image should be horizontally flipped else False
            :param resize_factor: factor by which to down sample the resolution, e.g 4 --> 1/4 resolution

            :return: Transformation composition
        """
        return transforms.Compose(
            [
                tf_prep.CV2ResizeByFactor(interpolation=cv2.INTER_NEAREST, factor=resize_factor),
                tf_prep.CV2HorizontalFlip(do_flip=do_flip),
                tf_prep.TransformToDepthSynthia(),
                tf_prep.PrepareForNet(do_normaliazion=False)
            ]
        )

    # ---------------------------functions below are useless but need to be defined--------------------------------------
    def transform_val(self, *args, **kwargs):
        pass

    @staticmethod
    def tf_rgb_val(*args, **kwargs):
        pass

    @staticmethod
    def tf_semantic_val(*args, **kwargs):
        pass

    @staticmethod
    def tf_depth_val(*args, **kwargs):
        pass


if __name__ == '__main__':
    from cfg.config_dataset import get_cfg_dataset_defaults
    path_16 = r'.\cfg\yaml_files\dataloaders\synthia_instances.yaml'
    cfg_16 = get_cfg_dataset_defaults()
    cfg_16.merge_from_file(path_16)
    cfg_16.freeze()

    dataset = SynthiaInstancesDataset('train', None, cfg_16)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    for idx, data in enumerate(loader):
        from utils.plotting_like_cityscapes_utils import visu_depth_prediction as vdp
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        data['rgb'][0] = data['rgb'][0] + torch.tensor([[[0.314747602, 0.277402550, 0.248091921]]]).transpose(0, 2)
        ax1.imshow(data['rgb'][0].numpy().transpose(1, 2, 0))
        print(data['rgb'][0].shape)
        ax2.imshow(data['semantic'][0])
        print(data['semantic'][0].shape)
        print(f'Min depth: {torch.min(data["depth_dense"])}')
        ax3.imshow(vdp(1 / data['depth_dense'][0]))
        print(data['depth_dense'][0].shape)
        plt.show()
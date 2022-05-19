# Project Imports
import torch
from torch import tensor

from dataloaders.dataset_base import DatasetSemantic, DatasetRGB, PathsHandlerSemantic
from misc import transforms as tf_prep

# Packages
import matplotlib.pyplot as plt
import random
import glob
import os
import PIL.Image as pil
import cv2
from torchvision import transforms

from utils.constans import IGNORE_INDEX_SEMANTIC


class _PathsSynthiaAugCityscapes(PathsHandlerSemantic):
    """
    Class for handling the paths to the images and labels.
    """

    def __init__(self, mode, _, cfg):
        """
        :param mode: mode of usage: 'train', 'val', 'test'
        :param split: split name of the data
        :param cfg: configuration
        """

        if mode != 'train':
            raise ValueError('SynthiaAugCityscapes dataset only supports \'train\' mode!')

        super(_PathsSynthiaAugCityscapes, self).__init__(mode, None, cfg)

        self.paths_unaugmented_rgb = self.get_unaugmented_rgb_image_paths(mode, split=None)

        if len(self.paths_unaugmented_rgb) == 0:
            assert False, "No Unaugmented RGB images could be found!"

    def get_rgb_image_paths(self, mode, frame_ids='*', file_format=".png", split=None):
        """
        Gets all the RGB image paths.
        :param mode: mode of usage: 'train', 'val', 'test'
        :param frame_ids: ids of frames to be selected, * = all images
        :param file_format: format of the files
        :param split: split name of the data
        """
        return sorted(
            glob.glob(
                self.get_rgb_image_path(self.path_base, frame_ids, file_format)
            )
        )

    def get_unaugmented_rgb_image_paths(self, mode, frame_ids='*', file_format=".png", split=None):
        """
        Gets all the RGB image paths.
        :param mode: mode of usage: 'train', 'val', 'test'
        :param frame_ids: ids of frames to be selected, * = all images
        :param file_format: format of the files
        :param split: split name of the data
        """
        return sorted(
            glob.glob(
                self.get_unaugmented_rgb_image_path(self.path_base, frame_ids, file_format)
            )
        )

    def get_semantic_label_paths(self, mode, frame_ids='*', file_format=".png", split=None):
        """
        Gets all the semantic label paths.
        :param mode: mode of usage: 'train', 'val', 'test'
        :param frame_ids: ids of frames to be selected, * = all images
        :param file_format: format of the files
        :param split: split name of the data
        """
        return sorted(
            glob.glob(
                self.get_semantic_label_path(self.path_base, frame_ids, file_format)
            )
        )

    @staticmethod
    def get_rgb_image_path(path_base, frame_ids, file_format):
        """
            Builds path to an rgb image.
            Default path: path_base/RGB/*.png
            Specific path: path_base/RGB/0000001.png
            :param path_base: path to directory containing images and labels
            :param frame_ids: 7-digit string id or '*'
            :param file_format: extensions of files '.png'
            :return: single path string
        """
        path = os.path.join(path_base, 'RGB', frame_ids + file_format)
        if os.path.exists(path) or not frame_ids.isnumeric():
            # Apply path exists check only if exact id is provided, no regex like '*'
            return path
        else:
            return None

    @staticmethod
    def get_unaugmented_rgb_image_path(path_base, frame_ids, file_format):
        """
            Builds path to an rgb image.
            Default path: path_base/RGB/*.png
            Specific path: path_base/RGB/0000001.png
            :param path_base: path to directory containing images and labels
            :param frame_ids: 7-digit string id or '*'
            :param file_format: extensions of files '.png'
            :return: single path string
        """
        path = os.path.join(path_base, 'RGB_UNAUG', frame_ids + file_format)
        if os.path.exists(path) or not frame_ids.isnumeric():
            # Apply path exists check only if exact id is provided, no regex like '*'
            return path
        else:
            return None

    @staticmethod
    def get_semantic_label_path(path_base, frame_ids, file_format):
        """
            Builds path to an semantic label image.
            Default path: path_base/GT/LABEL/*.png
            Specific path: path_base/GT/LABEL/0000001.png
            :param path_base: path to directory containing images and labels
            :param frame_ids: 7-digit string id or '*'
            :param file_format: extensions of files '.png'
            :return: single path string
        """
        path = os.path.join(path_base, 'LABELS', frame_ids + file_format)
        if os.path.exists(path) or not frame_ids.isnumeric():
            # Apply path exists check only if exact id is provided, no regex like '*'
            return path
        else:
            return None

    def paths_unaugmented_rgb(self):
        return self.paths_unaugmented_rgb


class SynthiaAugCityscapesDataset(DatasetRGB, DatasetSemantic):
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
        assert split is None, 'Synthia_Aug_Cityscapes currently does not support the spilt of the data!'
        assert mode == 'train', 'Synthia_Aug_Cityscapes does not support any other mode than train!'

        self.paths = _PathsSynthiaAugCityscapes(mode, split, cfg)

        super(SynthiaAugCityscapesDataset, self).__init__(pathsObj=self.paths, cfg=cfg)

        if self.cfg.dataset.num_classes == 16:

            self.class_names = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
                                "traffic_sign", "vegetation", "sky", "person", "rider", "car", "bus", "motorcycle",
                                "bicycle"]

            self.id_to_name = {i: name for i, name in enumerate(self.class_names)}
            self.valid_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        else:
            raise ValueError(f'Synthia not implemented for {self.cfg.dataset.num_classes} classes.')

        # will be assigned to void classes in the transformation, label for cityscapes pixels, that do not have a label
        self.ignore_index = IGNORE_INDEX_SEMANTIC

        # Stuff related to augmentations --------------------------------------------------------------

        self.aug_params = {"brightness_jitter": cfg.dataset.augmentation.brightness_jitter,
                           "contrast_jitter": cfg.dataset.augmentation.contrast_jitter,
                           "saturation_jitter": cfg.dataset.augmentation.saturation_jitter,
                           "hue_jitter": cfg.dataset.augmentation.hue_jitter}

        self.do_normalization = self.cfg.dataset.img_norm

        self.cityscapes_mean = torch.tensor([[0.28689554, 0.32513303, 0.28389177]]).transpose(1, 0)
        self.cityscapes_var = tensor([[1.0, 1.0, 1.0]]).transpose(1, 0)

        self.synthia_mean = torch.tensor([[0.314747602, 0.277402550, 0.248091921]]).transpose(1, 0)
        self.synthia_var = tensor([[1.0, 1.0, 1.0]]).transpose(1, 0)

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
        # Get all required training elements
        gt_semantic = self.get_semantic(
            self.paths.paths_semantic[index]) if self.paths.paths_semantic is not None else None

        rgb_imgs = {}

        rgb_imgs[0] = self.get_rgb(self.paths.paths_rgb[index], 0)  # not a sequence dataset therefor only offset 0

        unaug_rgb_imgs = {}  # unaugmented rgb images
        unaug_rgb_imgs[0] = self.get_unaug_rgb(self.paths.paths_unaugmented_rgb[index], 0)

        if self.mode == 'train':
            rgb_imgs, gt_semantic, unaug_rgb_imgs = self.transform_train(rgb_imgs, gt_semantic, unaug_rgb_imgs)
        elif self.mode == 'val' or self.mode == 'test':
            raise ValueError("Synthia only available for training")
        else:
            assert False, "The mode {} is not defined!".format(self.mode)

        # Create the map to be outputted
        data = {}

        if gt_semantic is not None:
            data["semantic"] = gt_semantic

        # normalize the rgb_img
        if self.do_normalization:
            rgb_imgs[0][:, data["semantic"] != 250] -= self.synthia_mean
            rgb_imgs[0][:, data["semantic"] == 250] -= self.cityscapes_mean
            unaug_rgb_imgs[0] -= self.cityscapes_mean.unsqueeze(1)

        data[("rgb", 0)] = rgb_imgs[0]
        data[("unaug_rgb", 0)] = unaug_rgb_imgs[0]

        return data

    def __str__(self):
        """
        Summarises and explains the dataset!
        :return: string describing the dataset
        """
        description = 'Valid Synthia Classes that have a matching valid cityscapes label:\n'
        description += 'Synthia_id: Class_name\n'
        for i, class_id in enumerate(self.valid_classes):
            description += f'{class_id}: {self.class_names[class_id]} \n'
        return description

    def transform_train(self, rgb_dict, gt_semantic, unaug_rgb_dict):
        """
            Transforms the rgb images and the semantic ground truth for training.
            :param unaug_rgb_dict: unaugmented rgb images (without synthia objects)
            :param gt_depth_dense: depth ground truth of image with offset = 0
            :param rgb_dict: dict of rgb images of a sequence.
            :param gt_semantic: semantic ground truth of the image with offset = 0
            :param gt_instance: instance labels of image with offset = 0
            :return: dict of transformed rgb images and transformed label
        """
        if self.cfg.dataset.debug:
            do_flip = False
            do_aug = False
        else:
            do_flip = random.random() > 0.5
            do_aug = random.random() > 0.5

        # Get the transformation objects
        tf_rgb_train = self.tf_rgb_train(tgt_size=self.feed_img_size,
                                         do_flip=do_flip,
                                         do_aug=do_aug,
                                         aug_params=self.aug_params)

        tf_unaug_rgb_train = self.tf_unaug_rgb_train(tgt_size=self.feed_img_size,
                                         do_flip=do_flip)

        tf_semantic_train = self.tf_semantic_train(tgt_size=self.feed_img_size,
                                                   do_flip=do_flip)

        # Apply transformations
        rgb_dict_tf = {}
        rgb_dict_tf[0] = tf_rgb_train(rgb_dict[0])  # not sequence dataset therefor only offset 0

        unaug_rgb_dict_tf = {}

        unaug_rgb_dict_tf[0] = tf_unaug_rgb_train(unaug_rgb_dict[0])  # not sequence dataset therefor only offset 0

        gt_semantic = tf_semantic_train(gt_semantic) if gt_semantic is not None else None

        return rgb_dict_tf, gt_semantic, unaug_rgb_dict_tf

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

    def get_unaug_rgb(self, path_file, *args):
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

    @staticmethod
    def tf_rgb_train(tgt_size, do_flip, do_aug, aug_params):
        """
        Transformations of the rgb image during training.
        :param aug_params: augmentation parameters
        :param do_aug: True if augmentations should be applied to the rgb image else False
        :param tgt_size: target size of the images after resize operation
        :param do_flip: True if the image should be horizontally flipped else False
        :param do_normalization: true if image should be normalized
        :param mean_city: mean of R,G,B of cityscapes rgb images
        :param var_city: variance in R,G,B of cityscapes rgb images
        :param mean_synthia: mean of R,G,B of synthia rgb images
        :param var_synthia: variance in R,G,B of synthia rgb images
        :param synthia_pixel_mask: boolean mask, true for synthia pixels, false for cityscapes pixels
        :return: Transformation composition
        """
        return transforms.Compose(
            [
                tf_prep.PILResize(tgt_size, pil.BICUBIC),
                tf_prep.PILHorizontalFlip(do_flip),
                tf_prep.ColorAug(do_aug, aug_params),
                tf_prep.PrepareForNet(do_normaliazion=False)
            ]
        )

    @staticmethod
    def tf_unaug_rgb_train(tgt_size, do_flip):
        """
        Transformations of the rgb image during training.
        :param aug_params: augmentation parameters
        :param do_aug: True if augmentations should be applied to the rgb image else False
        :param tgt_size: target size of the images after resize operation
        :param do_flip: True if the image should be horizontally flipped else False
        :param do_normalization: true if image should be normalized
        :param mean_city: mean of R,G,B of cityscapes rgb images
        :param var_city: variance in R,G,B of cityscapes rgb images
        :param mean_synthia: mean of R,G,B of synthia rgb images
        :param var_synthia: variance in R,G,B of synthia rgb images
        :param synthia_pixel_mask: boolean mask, true for synthia pixels, false for cityscapes pixels
        :return: Transformation composition
        """
        return transforms.Compose(
            [
                tf_prep.PILResize(tgt_size, pil.BICUBIC),
                tf_prep.PILHorizontalFlip(do_flip),
                tf_prep.PrepareForNet(do_normaliazion=False)
            ]
        )

    @staticmethod
    def tf_semantic_train(tgt_size, do_flip):
        """
            Transformations of the label during training.
            :param ignore_index: pixel will be labeled ignore_index if they are not part of a valid class
            :param s_to_c_mapping: dict mapping Synthia classes to valid Cityscapes classes
            :param valid_classes: list of valid classes
            :param void_classes: list of non valid classes (such pixel will be set to ignore index)
            :param tgt_size: target size of the labels after resize operation
            :param do_flip: True if the image should be horizontally flipped else False
            :return: Transformation composition
        """
        return transforms.Compose(
            [
                tf_prep.CV2Resize(tgt_size, interpolation=cv2.INTER_NEAREST),
                tf_prep.CV2HorizontalFlip(do_flip=do_flip),
                tf_prep.ToInt64Array(),
                tf_prep.ToTorchLong()
            ]
        )

    def get_valid_ids_and_names(self):
        """Returns valid training class ids and the corresponding class names."""
        names_ids = {}
        for idx, name in enumerate(self.class_names):
            if idx in self.valid_classes:
                names_ids[idx] = name
        names = [names_ids[key_id] for key_id in self.valid_classes]
        return self.valid_classes, names

    # ---------------------------functions below are useless but need to be defined--------------------------------------
    def transform_val(self, *args, **kwargs):
        pass

    @staticmethod
    def tf_rgb_val(*args, **kwargs):
        pass

    @staticmethod
    def tf_semantic_val(*args, **kwargs):
        pass


if __name__ == "__main__":
    torch.set_printoptions(precision=9)
    from cfg.config_dataset import get_cfg_dataset_defaults
    from torch.utils.data import DataLoader
    from utils.plotting_like_cityscapes_utils import semantic_id_tensor_to_rgb_numpy_array as s2rgb

    cfg = get_cfg_dataset_defaults()
    cfg.merge_from_file(
        r'C:\Users\benba\Documents\University\Masterarbeit\Depth-Semantic-UDA\cfg\yaml_files\dataloaders\synthia_aug_cityscapes.yaml')
    cfg.freeze()

    dataset = SynthiaAugCityscapesDataset("train", None, cfg)
    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False)

    for idx, data in enumerate(loader):
        print(idx)
        if idx != 738:
            continue
        semantic = data["semantic"].squeeze(0)
        rgb = data[("rgb", 0)].squeeze(0)
        unaug_rgb = data[("unaug_rgb", 0)].squeeze(0)
        rgb[:, semantic == 250] += dataset.cityscapes_mean
        rgb[:, semantic != 250] += dataset.synthia_mean
        unaug_rgb += dataset.cityscapes_mean.unsqueeze(1)
        print(torch.min(unaug_rgb))
        print(torch.max(unaug_rgb))
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(unaug_rgb.numpy().transpose(1, 2, 0))
        axs[1].imshow(rgb.numpy().transpose(1, 2, 0))
        axs[2].imshow(s2rgb(semantic.unsqueeze(0), num_classes=16))
        plt.show()

    print(dataset)

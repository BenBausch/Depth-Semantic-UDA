# Own files
from misc import transforms as tf_prep
from dataloaders import dataset_base
from cfg.config import get_cfg_defaults

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


class _PathsCityscapes(dataset_base.PathsHandlerSemantic):
    def __init__(self, mode, cfg):
        """
        :param mode: the mode ('train', 'val', 'test') in which the data is used.
        :param cfg: the configuration.
        """
        super(_PathsCityscapes, self).__init__(mode, None, cfg)

    def get_rgb_image_paths(self, mode, cities='*', frame_ids='*', seq_ids='*', data_type='*', file_format=".png", **kwargs):
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

    def get_semantic_label_paths(self, mode, cities='*', frame_ids='*', seq_ids='*', data_type='*', file_format=".png", **kwargs):
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
        path = os.path.join(path_base, 'images', mode, cities, cities + '_' + seq_ids + '_' + frame_ids + data_type +
                            file_format)
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

        path = os.path.join(path_base, 'labels', mode, cities, cities + '_' + seq_ids + '_' + frame_ids + data_type +
                            file_format)
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


class CityscapesSemanticDataset(dataset_base.DatasetSemantic):
    """
    Dataset for the semantically annotated images of the Cityscapes Dataset.
    """

    def __init__(self, mode, cfg):
        """
        Based on https://github.com/RogerZhangzz/CAG_UDA/blob/master/data/cityscapes_dataset.py
        Initializes the Cityscapes Semantic dataset by collecting all the paths and random shuffling the data if wanted.
        """
        self.paths = _PathsCityscapes(mode, cfg)
        super(CityscapesSemanticDataset, self).__init__(self.paths, cfg)

        self.cfg = cfg

        self.colors = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
                       [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60],
                       [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
        self.class_names = ["road","sidewalk","building","wall","fence","pole","traffic_light",
                            "traffic_sign","vegetation","terrain","sky","person","rider","car","truck",
                            "bus","train","motorcycle","bicycle"]
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.n_classes = 19
        self.label_colours = dict(zip(range(19), self.colors))
        self.class_map = dict(zip(self.valid_classes, range(19)))
        self.ignore_index = 250

        self.img_size = cfg.dataset.feed_img_size
        self.img_norm = cfg.dataset.img_norm

        self.mean_rgb = {
            "pascal": [103.939, 116.779, 123.68],
            "cityscapes": [0.0, 0.0, 0.0],
        }  # pascal mean for PSPNet and ICNet pre-trained model

        self.mean = np.array(self.mean_rgb['cityscapes'])

        self.ids = np.asarray([i for i in range(len(self.paths.paths_rgb))])

        if cfg.dataset.shuffle:
            np.random.shuffle(self.ids)

    def __len__(self):
        """
        :return: The number of ids in the split of the dataset.
        """
        return len(self.indexes)

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
        gt_semantic = self.get_semantic(self.paths.paths_semantic[index]) if self.paths.paths_semantic is not None else None

        rgb_imgs = {}
        for offset in self.rgb_frame_offsets:
            if self.get_rgb(self.paths.paths_rgb[index], offset) is not None:
                rgb_imgs[offset] = self.get_rgb(self.paths.paths_rgb[index], offset)
            else:  # As we can't train on the first and last image as temporal information is missing, we append the
                # dataset by two images for the beginning and the end by the corresponding non-offset RGB image
                rgb_imgs[offset] = self.get_rgb(self.paths.paths_rgb[index], 0)

        if self.mode == 'train':
            rgb_imgs, gt_semantic = self.transform_train(rgb_imgs, gt_semantic)
        elif self.mode == 'val' or self.mode =='test':
            rgb_imgs, sparse_depth, gt_depth = self.transform_val(rgb_imgs, gt_semantic)
        else:
            assert False, "The mode {} is not defined!".format(self.mode)

        # Create the map to be outputted
        data = {}

        if gt_semantic is not None:
            data["gt"] = gt_semantic

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
        tf_rgb_train = self.tf_rgb_train(self.feed_img_size, do_flip, self.mean)
        tf_semantic_train = self.tf_semantic_train(self.feed_img_size,
                                                   do_flip,
                                                   self.void_classes,
                                                   self.valid_classes,
                                                   self.class_map,
                                                   self.ignore_index)

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
        tf_rgb_val = self.tf_rgb_val(self.feed_img_size, self.mean)
        tf_semantic_val = self.tf_semantic_val(self.feed_img_size,
                                               self.void_classes,
                                               self.valid_classes,
                                               self.class_map,
                                               self.ignore_index)

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
            label = pil.open(path_file)
            return label

    def get_rgb(self, path_file, offset):
        """
        Loads the PIL Image for the path_file label and all the other rgb images with the given offsets in the sequence
        if they exist.
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

        img = pil.open(tgt_path_file)
        return img

    @staticmethod
    def tf_rgb_train(tgt_size, do_flip, mean):
        """
        Transformations of the rgb image during training.
        :param mean: mean of rgb images
        :param tgt_size: target size of the images after resize operation
        :param do_flip: True if the image should be horizontally flipped else False
        :return: Transformation composition
        """
        return transforms.Compose(
            [
                tf_prep.Resize(tgt_size, pil.BILINEAR),
                tf_prep.HorizontalFlip(do_flip),
                tf_prep.ToUint8Array(),
                tf_prep.NormalizeRGB(mean, True),
                tf_prep.PrepareForNet()
            ]
        )

    @staticmethod
    def tf_semantic_train(tgt_size, do_flip, void_classes, valid_classes, class_map, ignore_index):
        """
        Transformations of the label during training.
        :param ignore_index: pixel will be labeled ignore_index if they are not part of a valid class
        :param class_map: dict mapping valid class ids to numbers
        :param valid_classes: list of valid classes
        :param void_classes: list of non valid classes (such pixel will be set to ignore index)
        :param tgt_size: target size of the labels after resize operation
        :param do_flip: True if the image should be horizontally flipped else False
        :return: Transformation composition
        """
        return transforms.Compose(
            [
                tf_prep.Resize(tgt_size, pil.NEAREST),
                tf_prep.HorizontalFlip(do_flip),
                tf_prep.ToUint8Array(),
                tf_prep.EncodeSegmentation(void_classes, valid_classes, class_map, ignore_index)
            ]
        )

    @staticmethod
    def tf_rgb_val(tgt_size, mean):
        """
        Transformations of the rgb image during validation.
        :param mean: mean of rgb images
        :param tgt_size: target size of the images after resize operation
        :return: Transformation composition
        """
        return transforms.Compose(
            [
                tf_prep.Resize(tgt_size, pil.BILINEAR),
                tf_prep.ToUint8Array(),
                tf_prep.NormalizeRGB(mean, True),
                tf_prep.PrepareForNet()
            ]
        )

    @staticmethod
    def tf_semantic_val(tgt_size, void_classes, valid_classes, class_map, ignore_index):
        """
        Transformations of the labels during validation.
        :param ignore_index: pixel will be labeled ignore_index if they are not part of a valid class
        :param class_map: dict mapping valid class ids to numbers
        :param valid_classes: list of valid classes
        :param void_classes: list of non valid classes (such pixel will be set to ignore index)
        :param tgt_size: target size of the labels after resize operation
        :return: Transformation composition
        """
        return transforms.Compose(
            [
                tf_prep.Resize(tgt_size, pil.NEAREST),
                tf_prep.ToUint8Array(),
                tf_prep.EncodeSegmentation(void_classes, valid_classes, class_map, ignore_index)
            ]
        )

    def decode_segmap(self, temp):
        """
        Copied from https://github.com/RogerZhangzz/CAG_UDA/blob/master/data/gta5_dataset.py
        Decodes the class-id encoded segmentation map to an rgb encoded segmentation map.
        """
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file(
        r'C:\Users\benba\Documents\University\Masterarbeit\Depth-Semantic-UDA\cfg\train_cityscapes_semantic.yaml')
    cfg.eval.train.gt_depth_available = False
    cfg.eval.val.gt_depth_available = False
    cfg.eval.test.gt_depth_available = False
    cfg.dataset.use_sparse_depth = False
    cfg.eval.train.gt_semantic_available = True
    cfg.eval.val.gt_semantic_available = True
    cfg.eval.test.gt_semantic_available = True
    cfg.freeze()
    gta_dataset = CityscapesSemanticDataset('train', cfg)
    print((next(iter(gta_dataset))["gt"])[:,1,1])
    #plt.imshow((next(iter(gta_dataset))[("rgb", 0)].numpy().transpose(1, 2, 0)))
    #plt.show()
    plt.imshow((next(iter(gta_dataset))["gt"].numpy().transpose(1, 2, 0)))
    plt.show()

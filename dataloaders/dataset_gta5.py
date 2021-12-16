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
import ntpath

# Image processing
import PIL.Image as pil
from torchvision import transforms
import matplotlib.pyplot as plt

# Miscellaneous
import numbers
import random


def create_split_filter_file(splits, base_path, split):
    """
    Creates files with the ids in different files
    :param splits: split proposed by the authors of the GTA5 dataset
    :param base_path: the path to the base folder of the GTA5 dataset
    :param split: None: use all data for training, else force GTA5 split proposed by authors of the dataset.
    """
    if split is None:
        splits['trainIds'] = np.concatenate((splits['trainIds'], splits['valIds']))
        splits['trainIds'] = np.concatenate((splits['trainIds'], splits['testIds']))
        splits['trainIds'].sort(axis=0)
        splits['testIds'] = np.asarray([])
        splits['valIds'] = np.asarray([])

    split_names = ['trainIds', 'valIds', 'testIds']
    for split_name in split_names:
        path = os.path.join(base_path, split_name)
        if os.path.exists(path):
            os.remove(path)

        with open(path, 'w') as f:
            for id in splits[split_name]:
                f.write('{0:05d}'.format(id[0]) + "\n")


def check_ids_exist(path_base):
    """
    Removes the indexes of non existing files from the Ids of trainIds, valIds and testIds.
    :param path_base: path to the base directory of the GTA5 data
    """
    split_names = ['trainIds', 'valIds', 'testIds']
    for split_name in split_names:
        path = os.path.join(path_base, split_name)
        with open(path, 'r') as f_in:
            with open(path + '_temp', 'w') as f_out:
                for id in f_in:
                    if os.path.exists(os.path.join(path_base, 'images', id[:-1] + '.png')):
                        f_out.write(id)
        os.remove(path)
        os.rename(path + '_temp', path)


class _PathsGTA5(dataset_base.PathsHandlerSemantic):
    def __init__(self, mode, split, cfg):
        """
        Creates lists af all file_paths from the dataset.
        :param mode: the mode ('train', 'val', 'test') in which the data is used.
        :param split: If none use all data for training else force use split proposed by GTA5 dataset authors.
        :param cfg: the configuration.
        """
        splits = io.loadmat(os.path.join(cfg.dataset.path, 'split.mat')) # splits 24966 ids into 3 splits, even if not
        # all those images are present(not all downloaded)
        create_split_filter_file(splits, cfg.dataset.path, split)
        check_ids_exist(cfg.dataset.path)
        super(_PathsGTA5, self).__init__(mode, split, cfg)

    def get_rgb_image_paths(self, mode, frame_ids='*', file_format=".png", split=None):
        """
        Collects the paths to all rgb images.
        :param mode: 'train', 'val', 'test' are possible values
        :param frame_ids: '*' collect all images, or specific 5-digit string ids e.g. 00001
        :param file_format: extension of the files, '.png'
        :param split: which splitting method to use, None takes all data into data-loader, orther values enforce default
            split as introduced by authors of dataset (split.mat).
        :return: sorted list of files
        """
        if split is None: # i.e. no specific subset of files is given, hence use all data
            return sorted(
                glob.glob(
                    self.get_rgb_image_path(self.path_base, frame_ids, file_format)
                )
            )
        # use a subset of files given in a filter file
        else:
            filter_file = os.path.join(self.path_base, mode + "Ids")
            with open(filter_file, 'r') as f:
                files = [self.get_rgb_image_path(self.path_base, s, file_format) for s in f.read().splitlines()]
                files = np.array(files)[files != np.array(None)].tolist()
                return sorted(files)

    def get_semantic_label_paths(self, mode, frame_ids='*', file_format=".png", split=None):
        """
        Collects the paths to all semantic labels.
        :param mode: 'train', 'val', 'test' are possible values
        :param frame_ids: '*' collect all images, or specific 5-digit string ids e.g. 00001
        :param file_format: extension of the files, '.png'
        :param split: which splitting method to use, None takes all data into data-loader, orther values enforce default
            split as introduced by authors of dataset (split.mat).
        :return: sorted list of files
        """
        if split is None: # i.e. no specific subset of files is given, hence use all data
            return sorted(
                glob.glob(
                    self.get_semantic_label_path(self.path_base, frame_ids, file_format)
                )
            )
        # use a subset of files given in a filter file
        else:
            filter_file = os.path.join(self.path_base, mode + "Ids")
            with open(filter_file, 'r') as f:
                files = [self.get_semantic_label_path(self.path_base, s, file_format) for s in f.read().splitlines()]
                files = np.array(files)[files != np.array(None)].tolist()
                return sorted(files)

    @staticmethod
    def get_rgb_image_path(path_base, frame_ids, file_format):
        """
        Builds path to an rgb image.
        Default path: path_base/images/*.png
        Specific path: path_base/images/00001.png
        :param path_base: path to directory containing images and labels
        :param frame_ids: 5-digit string id or '*'
        :param file_format: extensions of files '.png'
        :return: single path string
        """
        path = os.path.join(path_base, 'images', frame_ids + file_format)
        if os.path.exists(path) or not frame_ids.isnumeric():
            # Apply check only if exact id is provided, no regex like '*'
            return path
        else:
            return None

    @staticmethod
    def get_semantic_label_path(path_base, frame_ids, file_format):
        """
        Builds path to an label.
        Default path: path_base/labels/*.png
        Specific path: path_base/labels/00001.png
        :param path_base: path to directory containing images and labels
        :param frame_ids: 5-digit string id or '*'
        :param file_format: extensions of files '.png'
        :return: single path string
        """
        path = os.path.join(path_base, 'labels', frame_ids + file_format)
        if os.path.exists(path) or not frame_ids.isnumeric():
            # Apply check only if exact id is provided, no regex like '*'
            return path
        else:
            return None


class GTA5Dataset(dataset_base.DatasetRGB, dataset_base.DatasetSemantic):
    def __init__(self, mode, split, cfg):
        """
        Based on https://github.com/RogerZhangzz/CAG_UDA/blob/master/data/gta5_dataset.py
        Initializes the GTA5 dataset by collecting all the paths and random shuffling the data if wanted.
        """
        self.paths = _PathsGTA5(mode, split, cfg)
        super(GTA5Dataset, self).__init__(self.paths, cfg)

        # dataset related values
        self.colors = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
                   [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60],
                   [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ["unlabelled", "road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
                            "traffic_sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus",
                            "train", "motorcycle", "bicycle"]
        self.n_classes = cfg.dataset.num_classes
        self.label_colours = dict(zip(range(19), self.colors))
        self.class_map = dict(zip(self.valid_classes, range(19)))
        self.ignore_index = 250

        self.mean = [0, 0, 0]

        self.split = split

        self.ids = []
        with open(os.path.join(self.paths.path_base, mode + 'Ids'), 'r') as f:
            for i in f.readlines():
                self.ids.append(int(i))
        self.ids = np.array(self.ids)

        if cfg.dataset.shuffle:
            np.random.shuffle(self.ids)

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
            rgb_imgs, gt_semantic = self.transform_val(rgb_imgs, gt_semantic)
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
        Loads the PIL Image with the given offset within the sequence form the path_file.

        :param path_file: path to the label
        :param offset: offsets of the other rgb images in the sequence to the path_file image.
        :return: PIL RGB Image.
        """
        assert isinstance(offset, numbers.Number), "The inputted offset {} is not numeric!".format(offset)
        assert os.path.exists(path_file), "The file {} does not exist!".format(path_file)

        tgt_path_file = path_file

        if offset != 0:
            tgt_frame_id = '{0:05d}'.format(int(path_file[-9:-4]) + offset)
            tgt_path_file = self.paths.get_rgb_image_path(path_file[:-17], tgt_frame_id, '.png')

        if tgt_path_file is None:
            return None

        img = pil.open(tgt_path_file)
        return img

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
                tf_prep.PILResize(tgt_size, pil.BILINEAR),
                tf_prep.PILHorizontalFlip(do_flip),
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
                tf_prep.PILResize(tgt_size, pil.NEAREST),
                tf_prep.PILHorizontalFlip(do_flip),
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
                tf_prep.PILResize(tgt_size, pil.BILINEAR),
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
                tf_prep.PILResize(tgt_size, pil.NEAREST),
                tf_prep.ToUint8Array(),
                tf_prep.EncodeSegmentation(void_classes, valid_classes, class_map, ignore_index)
            ]
        )


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file(
        r'C:\Users\benba\Documents\University\Masterarbeit\Depth-Semantic-UDA\cfg\train_gta5_semantic.yaml')
    cfg.eval.train.gt_depth_available = False
    cfg.eval.val.gt_depth_available = False
    cfg.eval.test.gt_depth_available = False
    cfg.dataset.use_sparse_depth = False
    cfg.eval.train.gt_semantic_available = True
    cfg.eval.val.gt_semantic_available = True
    cfg.eval.test.gt_semantic_available = True
    cfg.freeze()
    gta_dataset = GTA5Dataset('train', 'train', cfg)
    #plt.imshow((next(iter(gta_dataset))[("rgb", 0)].numpy().transpose(1, 2, 0)))
    plt.show()
    #plt.imshow((next(iter(gta_dataset))[("rgb", 1)].numpy().transpose(1, 2, 0)))
    plt.show()
    #plt.imshow((next(iter(gta_dataset))[("rgb", -1)].numpy().transpose(1, 2, 0)))
    plt.show()
    print((next(iter(gta_dataset))["gt"]))
    plt.imshow((next(iter(gta_dataset))["gt"]))
    plt.show()

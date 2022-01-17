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


# fixme: if split is None, the whole dataset will be used no matter the mode 'train' or 'val', is this an appropriate
#  behaviour
class _PathsSynthiaRandCityscapes(PathsHandlerSemantic, PathsHandlerDepthDense):

    def __init__(self, mode, split, cfg):

        if split is not None and not split in available_splits.keys():
            raise Exception('Non existing split selected!')

        super(_PathsSynthiaRandCityscapes, self).__init__(mode, split, cfg)

        # Check if dataset has been prepared before use
        if not os.path.exists(os.path.join(self.path_base, 'splits')):
            raise Exception(f'Please run the python script '
                            f'{os.path.join("dataloaders", "scripts", "prepare_synthia.py")}'
                            f'script before using the Synthia dataset!')

    def get_rgb_image_paths(self, mode, frame_ids='*', file_format=".png", split=None):
        if split is None:  # i.e. no specific subset of files is given, hence use all data
            return sorted(
                glob.glob(
                    self.get_rgb_image_path(self.path_base, frame_ids, file_format)
                )
            )
        # use a subset of files given in a filter file
        else:
            filter_file = os.path.join(self.path_base, "splits", split, mode + ".txt")
            with open(filter_file, 'r') as f:
                return sorted(
                    [self.get_rgb_image_path(self.path_base, frame_ids=s,
                                             file_format=file_format) for s in f.read().splitlines()]
                )

    def get_semantic_label_paths(self, mode, frame_ids='*', file_format=".png", split=None):
        if split is None:  # i.e. no specific subset of files is given, hence use all data
            return sorted(
                glob.glob(
                    self.get_semantic_label_path(self.path_base, frame_ids, file_format)
                )
            )
        # use a subset of files given in a filter file
        else:
            filter_file = os.path.join(self.path_base, "splits", split, mode + ".txt")
            with open(filter_file, 'r') as f:
                return sorted(
                    [self.get_semantic_label_path(self.path_base, frame_ids=s,
                                                  file_format=file_format) for s in f.read().splitlines()]
                )

    def get_gt_depth_image_paths(self, mode, frame_ids='*', file_format=".png", split=None):
        if split is None:  # i.e. no specific subset of files is given, hence use all data
            return sorted(
                glob.glob(
                    self.get_depth_label_path(self.path_base, frame_ids, file_format)
                )
            )
        # use a subset of files given in a filter file
        else:
            filter_file = os.path.join(self.path_base, "splits", split, mode + ".txt")
            with open(filter_file, 'r') as f:
                return sorted(
                    [self.get_depth_label_path(self.path_base, frame_ids=s,
                                               file_format=file_format) for s in f.read().splitlines()]
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
        path = os.path.join(path_base, 'GT', 'LABELS', frame_ids + file_format)
        if os.path.exists(path) or not frame_ids.isnumeric():
            # Apply path exists check only if exact id is provided, no regex like '*'
            return path
        else:
            return None

    @staticmethod
    def get_depth_label_path(path_base, frame_ids, file_format):
        """
            Builds path to an rgb image.
            Default path: path_base/RGB/*.png
            Specific path: path_base/RGB/0000001.png
            :param path_base: path to directory containing images and labels
            :param frame_ids: 7-digit string id or '*'
            :param file_format: extensions of files '.png'
            :return: single path string
        """
        path = os.path.join(path_base, 'Depth', 'Depth', frame_ids + file_format)
        if os.path.exists(path) or not frame_ids.isnumeric():
            # Apply path exists check only if exact id is provided, no regex like '*'
            return path
        else:
            return None


class SynthiaRandCityscapesDataset(DatasetRGB, DatasetSemantic, DatasetDepth):
    """
        RGB Images annotated with Depth and Semantic classes. This dataset does not provide any image sequences.
    """

    def __init__(self, mode, split, cfg):
        # The dataset does not contain sequences therefore the offsets refer only to the selected RGB image(offset == 0)
        assert len(cfg.dataset.rgb_frame_offsets) == 1
        assert cfg.dataset.rgb_frame_offsets[0] == 0

        self.paths = _PathsSynthiaRandCityscapes(mode, split, cfg)

        super(SynthiaRandCityscapesDataset, self).__init__(pathsObj=self.paths, cfg=cfg)

        # Stuff realted to class encoding -------------------------------------------------------------
        # class name index in list corresponds to id of label
        self.class_names = ['void', 'sky', 'Building', 'Road', 'Sidewalk', 'Fence', 'Vegetation', 'Pole', 'Car',
                            'Traffic_sign', 'Pedestrian', 'Bicycle', 'Motorcycle', 'Parking-slot', 'Road-work',
                            'Traffic_light', 'Terrain', 'Rider', 'Truck', 'Bus', 'Train', 'Wall', 'Lanemarking']

        self.synthia_id_to_synthia_name = {i: name for i, name in enumerate(self.class_names)}

        # classes considered void have no equivalent valid label in cityscapes
        # See: https://www.cityscapes-dataset.com/dataset-overview/ for void and valid classes
        self.valid_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 17, 18, 19, 20, 21]
        self.void_classes = [0, 13, 14, 22]

        # cityscapes maps the ids of all the valid classes back to ids ranging from 0 to 18
        # (road initially labeled 7 remapped to label 0)
        # all other classes are considered void classes (pole group, guard rail, ...)
        # when talking about cityscapes label, we refer to the remapped labels
        self.synthia_id_to_cityscapes_id = {1: 10, 2: 2, 3: 0, 4: 1, 5: 4, 6: 8, 7: 5, 8: 13, 9: 7,
                                            10: 11, 11: 18, 12: 17, 15: 6, 16: 9,
                                            17: 12, 18: 14, 19: 15, 20: 16, 21: 3}

        self.ignore_index = 250  # will be assigned to void classes in the transformation

        # Stuff realted to augmentations --------------------------------------------------------------

        self.aug_params = {"brightness_jitter": cfg.dataset.augmentation.brightness_jitter,
                           "contrast_jitter": cfg.dataset.augmentation.contrast_jitter,
                           "saturation_jitter": cfg.dataset.augmentation.saturation_jitter,
                           "hue_jitter": cfg.dataset.augmentation.hue_jitter}

        self.do_normalization = self.cfg.dataset.img_norm

        # todo: validate these values
        self.mean = torch.tensor([[[0.314747602, 0.277402550, 0.248091921]]]).transpose(0, 2)
        self.var = tensor([[[0.073771872, 0.062956616, 0.062426906]]]).transpose(0, 2)

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

        gt_depth_dense = self.get_depth(self.paths.paths_depth_gt[index]) if self.paths.paths_depth_gt is not None \
            else None

        rgb_imgs = {}
        for offset in self.rgb_frame_offsets:
            if self.get_rgb(self.paths.paths_rgb[index], offset) is not None:
                rgb_imgs[offset] = self.get_rgb(self.paths.paths_rgb[index], offset)
            else:  # As we can't train on the first and last image as temporal information is missing, we append the
                # dataset by two images for the beginning and the end by the corresponding non-offset RGB image
                rgb_imgs[offset] = self.get_rgb(self.paths.paths_rgb[index], 0)

        if self.mode == 'train':
            rgb_imgs, gt_semantic, gt_depth_dense = self.transform_train(rgb_imgs, gt_semantic, gt_depth_dense)
        elif self.mode == 'val' or self.mode == 'test':
            rgb_imgs, gt_semantic, gt_depth_dense = self.transform_val(rgb_imgs, gt_semantic, gt_depth_dense)
        else:
            assert False, "The mode {} is not defined!".format(self.mode)

        # Create the map to be outputted
        data = {}

        if gt_semantic is not None:
            data["semantic"] = gt_semantic

        if gt_depth_dense is not None:
            data["depth_dense"] = gt_depth_dense

        for offset, val in rgb_imgs.items():
            data[("rgb", offset)] = val

        return data

    def __str__(self):
        """
        Summarises and explains the dataset!
        :return: string describing the dataset
        """
        description = 'Valid Synthia Classes that have a matching valid cityscapes label:\n'
        description += 'Synthia_id: Class_name\n'
        for i, class_id in enumerate(self.valid_classes):
            if class_id not in self.void_classes:
                description += f'{class_id}: {self.class_names[class_id]} \n'
        description += '\n\n'
        description += 'Void Synthia Classes that do not have a matching valid cityscapes label:\n'
        description += 'Synthia_id: Class_name\n'
        for i, class_id in enumerate(self.void_classes):
            description += f'{class_id}: {self.class_names[class_id]} \n'
        description += '\n\n'
        description += 'Mapping of the Synthia ids to the Cityscapes ids:\n'
        description += 'Synthia_id: Cityscape_id\n'
        for synthia_id in self.synthia_id_to_cityscapes_id.keys():
            description += f'{synthia_id}: {self.synthia_id_to_cityscapes_id[synthia_id]}\n'
        description += '\n'

        return description

    def transform_train(self, rgb_dict, gt_semantic, gt_depth_dense):
        """
            Transforms the rgb images and the semantic ground truth for training.
            :param gt_depth_dense: depth ground truth of image with offset = 0
            :param rgb_dict: dict of rgb images of a sequence.
            :param gt_semantic: semantic ground truth of the image with offset = 0
            :return: dict of transformed rgb images and transformed label
        """
        do_flip = random.random() > 0.5
        do_aug = random.random() > 0.5

        # Get the transformation objects
        tf_rgb_train = self.tf_rgb_train(tgt_size=self.feed_img_size,
                                         do_flip=do_flip,
                                         do_aug=do_aug,
                                         aug_params=self.aug_params,
                                         do_normalization=self.do_normalization,
                                         mean=self.mean,
                                         var=self.var)
        tf_semantic_train = self.tf_semantic_train(tgt_size=self.feed_img_size,
                                                   do_flip=do_flip,
                                                   s_to_c_mapping=self.synthia_id_to_cityscapes_id,
                                                   valid_classes=self.valid_classes,
                                                   void_classes=self.void_classes,
                                                   ignore_index=self.ignore_index)
        tf_depth_dense_train = self.tf_depth_train(tgt_size=self.feed_img_size,
                                                   do_flip=do_flip)

        # Apply transformations
        rgb_dict_tf = {}
        for k, img in rgb_dict.items():
            rgb_dict_tf[k] = tf_rgb_train(img) if img is not None else None

        gt_semantic = tf_semantic_train(gt_semantic) if gt_semantic is not None else None

        gt_depth_dense = tf_depth_dense_train(gt_depth_dense) if gt_depth_dense is not None else None

        return rgb_dict_tf, gt_semantic, gt_depth_dense

    def transform_val(self, rgb_dict, gt_semantic, gt_depth_dense):
        """
            Transforms the rgb images and the semantic ground truth for validation.
            :param gt_depth_dense: depth ground truth of the image with offset = 0
            :param rgb_dict: dict of rgb images of a sequence, for this dataset it should be a one image sequence.
            :param gt_semantic: semantic ground truth of the image with offset = 0
            :return: dict of transformed rgb images and transformed label
        """
        # Get the transformation objects
        tf_rgb_val = self.tf_rgb_val(tgt_size=self.feed_img_size,
                                     do_normalization=self.do_normalization,
                                     mean=self.mean,
                                     var=self.var)
        tf_semantic_val = self.tf_semantic_val(tgt_size=self.feed_img_size,
                                               s_to_c_mapping=self.synthia_id_to_cityscapes_id,
                                               valid_classes=self.valid_classes,
                                               void_classes=self.void_classes,
                                               ignore_index=self.ignore_index)

        tf_depth_dense_val = self.tf_depth_val(tgt_size=self.feed_img_size)

        # Apply transformations
        rgb_dict_tf = {}
        for k, img in rgb_dict.items():
            rgb_dict_tf[k] = tf_rgb_val(img) if img is not None else None

        gt_semantic = tf_semantic_val(gt_semantic) if gt_semantic is not None else None

        gt_depth_dense = tf_depth_dense_val(gt_depth_dense) if gt_depth_dense is not None else None

        return rgb_dict_tf, gt_semantic, gt_depth_dense

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

            img = pil.open(path_file)
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

            label = cv2.imread(path_file, cv2.IMREAD_UNCHANGED)[:, :, 2]
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

            label = cv2.imread(path_file, cv2.IMREAD_UNCHANGED)[:, :, 0]
            return label

    @staticmethod
    def tf_rgb_train(tgt_size, do_flip, do_aug, do_normalization, aug_params, mean, var):
        """
        Transformations of the rgb image during training.
        :param aug_params: augmentation parameters
        :param do_aug: True if augmentations should be applied to the rgb image else False
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
                tf_prep.ColorAug(do_aug, aug_params),
                tf_prep.PrepareForNet(do_normalization, mean, var)
            ]
        )

    @staticmethod
    def tf_semantic_train(tgt_size, do_flip, s_to_c_mapping, valid_classes, void_classes, ignore_index):
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
                tf_prep.Syntia_To_Cityscapes_Encoding(s_to_c_mapping=s_to_c_mapping,
                                                      valid_classes=valid_classes,
                                                      void_classes=void_classes,
                                                      ignore_index=ignore_index)
            ]
        )

    @staticmethod
    def tf_depth_train(tgt_size, do_flip):
        return transforms.Compose(
            [
                tf_prep.CV2Resize(tgt_size, interpolation=cv2.INTER_NEAREST),
                tf_prep.CV2HorizontalFlip(do_flip=do_flip),
                tf_prep.TransformToDepthSynthia(),
                tf_prep.tf.ToTensor(),
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
    def tf_semantic_val(tgt_size, s_to_c_mapping, valid_classes, void_classes, ignore_index):
        """
            Transformations of the label during training.
            :param ignore_index: pixel will be labeled ignore_index if they are not part of a valid class
            :param s_to_c_mapping: dict mapping Synthia classes to valid Cityscapes classes
            :param valid_classes: list of valid classes
            :param void_classes: list of non valid classes (such pixel will be set to ignore index)
            :param tgt_size: target size of the labels after resize operation
            :return: Transformation composition
        """
        return transforms.Compose(
            [
                tf_prep.CV2Resize(tgt_size, interpolation=cv2.INTER_NEAREST),
                tf_prep.ToInt64Array(),
                tf_prep.Syntia_To_Cityscapes_Encoding(s_to_c_mapping=s_to_c_mapping,
                                                      valid_classes=valid_classes,
                                                      void_classes=void_classes,
                                                      ignore_index=ignore_index)
            ]
        )

    @staticmethod
    def tf_depth_val(tgt_size):
        return transforms.Compose(
            [
                tf_prep.CV2Resize(tgt_size, interpolation=cv2.INTER_NEAREST),
                tf_prep.TransformToDepthSynthia(),
                tf_prep.tf.ToTensor()
            ]
        )


if __name__ == "__main__":
    from cfg.config_dataset import get_cfg_dataset_defaults

    cfg = get_cfg_dataset_defaults()
    cfg.merge_from_file(
        r'C:\Users\benba\Documents\University\Masterarbeit\Depth-Semantic-UDA\cfg\yaml_files\train\guda\synthia.yaml')
    cfg.freeze()

    img_h = cfg.dataset.feed_img_size[1]
    img_w = cfg.dataset.feed_img_size[0]

    ds = SynthiaRandCityscapesDataset(mode='val', split=None, cfg=cfg)
    batch_size = 1
    ds = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    import wandb
    from utils.plotting_utils import CITYSCAPES_ID_TO_NAME, CITYSCAPES_CLASS_IDS, CITYSCAPES_CLASS_NAMES, \
    CITYSCAPES_ID_TO_COLOR
    from utils.plotting_utils import semantic_id_tensor_to_rgb_numpy_array as s_to_rgb
    import matplotlib.patches as mpatches
    import matplotlib.colors as colors

    patches = [
        mpatches.Patch(color=colors.to_rgba(CITYSCAPES_ID_TO_COLOR[i]/255), label=f'{i}. {CITYSCAPES_CLASS_NAMES[i]}') for
        i in
        range(len(CITYSCAPES_CLASS_IDS) - 1)]
    patches.append(
        mpatches.Patch(color=colors.to_rgba(CITYSCAPES_ID_TO_COLOR[250]/255), label=f'{250}. {CITYSCAPES_CLASS_NAMES[-1]}'))

    wandb.init(project='synthia')

    torch.set_printoptions(precision=9)

    for i, data in enumerate(ds):
        img0 = wandb.Image(data[('rgb', 0)].squeeze(0).numpy().transpose(1, 2, 0), caption="RGB")
        img_semantic = wandb.Image(data[('rgb', 0)].squeeze(0).numpy().transpose(1, 2, 0),
                                   masks={'ground_truth': {
                                       'mask_data': data['semantic'].squeeze(0).numpy(),
                                       'class_labels': CITYSCAPES_ID_TO_NAME
                                   }}, caption="Semantic")
        img_depth = wandb.Image(data['depth_dense'].squeeze(0).numpy().transpose(1, 2, 0), caption="Depth")
        wandb.log({'images': [img0, img_semantic, img_depth]})
        plt.imshow(s_to_rgb(data['semantic']))
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        if i == 10:
            break

    """mean = torch.tensor([0.0, 0.0, 0.0]).to('cuda:0')
    for i, data in enumerate(ds):
        #print(i)
        data = data[('rgb', 0)].to('cuda:0')
        data = data.view(data.size(0), data.size(1), -1)
        mean += data.mean(2).sum(0) / batch_size
    print(mean / len(ds))

    print('Calculating var:')
    var = torch.tensor([0.0, 0.0, 0.0]).to('cuda:0')
    for i, data in enumerate(ds):
        #print(i)
        data = data[('rgb', 0)].to('cuda:0')
        # data = data.view(data.size(0), data.size(1), -1)
        var += torch.sum(((data - torch.tensor([[[0.314747602, 0.277402550, 0.248091921]]]).transpose(0, 2)) ** 2), dim=(0, 2, 3)) / (batch_size * img_h * img_w)
        print(var / (i + 1))
    var = var / len(ds)"""
    r"""data = next(iter(ds))
    for i, data in enumerate(ds):
        if i == 10:
            break
        data = data
    plt.subplot(2, 2, 1)
    plt.imshow(data[('rgb', 0)].numpy().transpose(1, 2, 0))
    plt.subplot(2, 2, 2)

    from utils.plotting_utils import CITYSCAPES_CLASS_NAMES,CITYSCAPES_ID_TO_COLOR,CITYSCAPES_CLASS_IDS,\
        cityscapes_cmap_norm, cityscapes_cmap
    import matplotlib.patches as mpatches
    import matplotlib.colors as colors

    patches = [mpatches.Patch(color=colors.to_rgba(CITYSCAPES_ID_TO_COLOR[i]), label=f'{i}. {CITYSCAPES_CLASS_NAMES[i]}') for
               i in
               range(len(CITYSCAPES_CLASS_IDS) - 1)]
    patches.append(
        mpatches.Patch(color=colors.to_rgba(CITYSCAPES_ID_TO_COLOR[250]), label=f'{250}. {CITYSCAPES_CLASS_NAMES[-1]}'))

    im = plt.imshow(data['semantic'], cmap=cityscapes_cmap, norm=cityscapes_cmap_norm)

    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.subplot(2, 2, 3)

    file = r'/data/SYNTHIA_RAND_CITYSCAPES/RAND_CITYSCAPES/GT/LABELS/0000631.png'

    import torch
    print(torch.max(data['depth_dense']))
    plt.imshow(data['depth_dense'].squeeze(0))
    plt.colorbar()

    plt.show()
    semantic = ds.get_semantic(file)
    plt.subplot(1, 2, 1)
    plt.imshow(semantic)
    print(semantic.shape)
    tran = ds.tf_semantic_train(tgt_size=cfg.dataset.feed_img_size,
                                do_flip=False,
                                s_to_c_mapping=ds.synthia_id_to_cityscapes_id,
                                valid_classes=ds.valid_classes,
                                void_classes=ds.void_classes,
                                ignore_index=ds.ignore_index)
    semantic = tran(semantic)
    plt.subplot(1, 2, 2)
    plt.imshow(semantic)
    plt.show()
    print(semantic.shape)"""
    r"""file = r'C:\Users\benba\Documents\University\Masterarbeit\data\SYNTHIA_RAND_CITYSCAPES\RAND_CITYSCAPES\Depth\Depth\0000638.png'
    d = np.array(pil.open(file).convert('L'))
    print(d)"""

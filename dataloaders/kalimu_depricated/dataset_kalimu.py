# Own files
from misc import transforms as tf_prep
from dataloaders import dataset_base

# External libraries
# I/O
import os
import os.path
import glob
import ntpath

# Image processing
import PIL.Image as pil
from torchvision import transforms

# Miscellaneous
import numbers
import random


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
# guaranteed here. If the kalimu_depricated dataset is sorted by the scripts as in this framework, then the correct order is also guaranteed.
class _PathsKalimuFromMotion(dataset_base.PathsHandlerDepthFromMotion):
    def __init__(self, mode, split, cfg):
        super(_PathsKalimuFromMotion, self).__init__(mode, split, cfg)

    def get_rgb_image_paths(self, mode, drive_seqs='*', cam_ids='image_blackfly', frame_ids='*', format=".png", split=None):
        if split is None: # i.e. no specific subset of files is given, hence use all data
            a = self.get_rgb_image_path(self.path_base, mode, drive_seqs, cam_ids, frame_ids, format)
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

    def get_gt_depth_image_paths(self, mode, drive_seqs='*', cam_ids='image_blackfly', frame_ids='*', format=".png", split=None):
        if not self.gt_depth_available:
            return None
        else:
            if split is None:
                return sorted(
                    glob.glob(
                        self.get_gt_depth_image_path(self.path_base, mode, drive_seqs, cam_ids, frame_ids, format)
                    )
                )
            else:
                filter_file = '{}/{}/{}'.format(self.path_base, "splits", split+"_"+mode+".txt")
                with open(filter_file, 'r') as f:
                    return sorted(
                        [self.get_gt_depth_image_path(self.path_base, mode, s.split()[0], s.split()[1][-1], s.split()[2], s.split()[3]) for s in f.read().splitlines()]
                    )

    def get_sparse_depth_image_paths(self, mode, drive_seqs='*', cam_ids='image_blackfly', frame_ids='*', format=".png", split=None):
        if not self.use_sparse_depth:
            return None
        else:
            if split is None:
                return sorted(
                    glob.glob(
                        self.get_sparse_depth_image_path(self.path_base, mode, drive_seqs, cam_ids, frame_ids, format)
                    )
                )
            else:
                filter_file = '{}/{}/{}'.format(self.path_base, "splits", split+"_"+mode+".txt")
                with open(filter_file, 'r') as f:
                    return sorted(
                        [self.get_sparse_depth_image_path(self.path_base, mode, s.split()[0], s.split()[1][-1], s.split()[2], s.split()[3]) for s in f.read().splitlines()]
                    )

    @staticmethod
    def get_rgb_image_path(path_base, mode, drive_seqs, cam_ids, frame_ids, format):
        return '{}/{}/{}/rgb/{}/data/{}{}'.format(path_base, mode, drive_seqs, cam_ids, frame_ids, format)

    @staticmethod
    def get_gt_depth_image_path(path_base, mode, drive_seqs, cam_ids, frame_ids, format):
        return '{}/{}/{}/groundtruth/proj_depth/{}/{}{}'.format(path_base, mode, drive_seqs, cam_ids, frame_ids, format)

    @staticmethod
    def get_sparse_depth_image_path(path_base, mode, drive_seqs, cam_ids, frame_ids, format):
        return '{}/{}/{}/sick_raw/proj_depth/{}/{}{}'.format(path_base, mode, drive_seqs, cam_ids, frame_ids, format)

    @staticmethod
    def get_calib_path(path_base):
        return '{}/calib/calib.txt'.format(path_base)

    # We assume that there is an RGB image for each depth image but not vice versa. Apart from that it is assumed
    # that the depth images are also consistent if downloaded from the official Kalimu webpage
    # So for now we just focus on deleting those rgb images that have no corresponding depth image as this should be
    # enough to have a fully consistent dataset
    def make_consistent(self):
        # If neither sparse nor dense GT depth maps are to be used, then the dataset is per definition consistent, i.e.
        # you just use the RGB images that are available
        if not self.use_sparse_depth and not self.gt_depth_available:
            return True
        else:
            # Check if all of the specified rgb paths do exist
            path_rgb_to_be_deleted = []
            for path_rgb in self.paths_rgb:
                if not os.path.exists(path_rgb):
                    path_rgb_to_be_deleted.append(path_rgb)
            for path in path_rgb_to_be_deleted:
                self.paths_rgb.remove(path)

            # Check if all of the specified sparse depth paths do exist
            if self.use_sparse_depth:
                path_sparse_to_be_deleted = []
                for path_sparse in self.paths_depth_sparse:
                    if not os.path.exists(path_sparse):
                        path_sparse_to_be_deleted.append(path_sparse)
                for path in path_sparse_to_be_deleted:
                    self.paths_depth_sparse.remove(path)

            # Check if all of the specified gt depth paths do exist
            if self.gt_depth_available:
                path_gt_to_be_deleted = []
                for path_gt in self.paths_depth_gt:
                    if not os.path.exists(path_gt):
                        path_gt_to_be_deleted.append(path_gt)
                for path in path_gt_to_be_deleted:
                    self.paths_depth_gt.remove(path)

            path_rgb_to_be_deleted = []
            if self.use_sparse_depth:
                paths_depth_dict = dict.fromkeys(self.paths_depth_sparse, 1)
                get_path = self.get_sparse_depth_image_path
            else:
                paths_depth_dict = dict.fromkeys(self.paths_depth_gt, 1)
                get_path = self.get_gt_depth_image_path

            # Get the depth map formats first as they remain the same
            if self.use_sparse_depth:
                format_sparse_depth = os.path.splitext(path_leaf(self.paths_depth_sparse[0]))[1]
            if self.gt_depth_available:
                format_gt = os.path.splitext(path_leaf(self.paths_depth_gt[0]))[1]

            # Get all candidate sparse depth and GT dense depth maps from rgb image paths
            for path_rgb in self.paths_rgb:
                # Get essential parts of the rgb path to find the corresponding depth image
                # ToDo: Decouple depth file_format from image file_format
                mode, drive_seq, cam_id, frame_id, img_format = decompose_rgb_path(path_rgb)

                # Only sparse depth are gone through as sparse and dense depth data are consistent
                candidate_path = get_path(self.path_base, mode, drive_seq, cam_id, frame_id, img_format)

                if not os.path.exists(candidate_path) or candidate_path not in paths_depth_dict:
                    path_rgb_to_be_deleted.append(path_rgb)

            for path in path_rgb_to_be_deleted:
                self.paths_rgb.remove(path)

    # Go through the rgb dataset, decompose it and look whether the corresponding file exists at the exactly same
    # position in the sparse and gt path lists... This makes sure that the lists are fully consistent
    def is_consistent(self):
        if not self.use_sparse_depth and not self.gt_depth_available:
            return True

        # Get the depth map formats first as they remain the same
        format_sparse_depth = ""
        format_gt = ""
        if self.use_sparse_depth:
            format_sparse_depth = os.path.splitext(path_leaf(self.paths_depth_sparse[0]))[1]
        if self.gt_depth_available:
            format_gt = os.path.splitext(path_leaf(self.paths_depth_gt[0]))[1]

        for i in range(len(self.paths_rgb)):
            # Attention: We assume that both the RGB images and depth maps have the same file_format (.png), which is
            # actually the case for the Kalimu dataset
            mode_rgb, drive_seq_rgb, cam_id_rgb, frame_id_rgb, _ = decompose_rgb_path(self.paths_rgb[i])

            if self.gt_depth_available:
                candidate_path_gt = self.get_gt_depth_image_path(self.path_base, mode_rgb, drive_seq_rgb, cam_id_rgb, frame_id_rgb, format_gt)
                if candidate_path_gt != self.paths_depth_gt[i]:
                    return False
            if self.use_sparse_depth:
                candidate_path_sparse = self.get_sparse_depth_image_path(self.path_base, mode_rgb, drive_seq_rgb, cam_id_rgb, frame_id_rgb, format_sparse_depth)
                if candidate_path_sparse != self.paths_depth_sparse[i]:
                    return False
        return True

# Note: We set all parameters that we need out of the cfg object inside of the constructor for the sake of readability.
# So the reader knows which arguments inside of the cfg object are actually used...
class KalimuDataset(dataset_base.DatasetRGB, dataset_base.DatasetDepth):
    def __init__(self, mode, split, cfg):
        kalimuPaths = _PathsKalimuFromMotion(mode, split, cfg)
        super(KalimuDataset, self).__init__(kalimuPaths, cfg)

        # Initialize the paths here
        self.feed_img_size = cfg.dataset.feed_img_size
        self.aug_params = {"brightness_jitter": cfg.dataset.augmentation.brightness_jitter,
                           "contrast_jitter": cfg.dataset.augmentation.contrast_jitter,
                           "saturation_jitter": cfg.dataset.augmentation.saturation_jitter,
                           "hue_jitter": cfg.dataset.augmentation.hue_jitter}

    # Here the input takes a whole dictionary of RGB images, as for training multiple RGB images may be used
    def transform_train(self, rgb_dict, sparse_depth, gt_depth):
        # Specify randomly whether a horizontal flip should be done
        do_flip = random.random() > 0.5
        do_aug = random.random() > 0.5

        # Get the transformation objects
        tf_rgb_train = self.tf_rgb_train(self.feed_img_size, do_flip, do_aug, self.aug_params)
        tf_depth_train = self.tf_depth_train(self.feed_img_size, do_flip)

        rgb_dict_tf = {}
        for k, img in rgb_dict.items():
            rgb_dict_tf[k] = tf_rgb_train(img) if img is not None else None

        sparse_depth = tf_depth_train(sparse_depth) if sparse_depth is not None else None
        gt_depth = tf_depth_train(gt_depth) if gt_depth is not None else None

        return rgb_dict_tf, sparse_depth, gt_depth

    def transform_val(self, rgb_dict, sparse_depth, gt_depth):
        tf_rgb_val = self.tf_rgb_val(self.feed_img_size)
        tf_depth_val = self.tf_depth_val(self.feed_img_size)

        rgb_dict_tf = {}
        for k, img in rgb_dict.items():
            rgb_dict_tf[k] = tf_rgb_val(img) if img is not None else None

        sparse_depth = tf_depth_val(sparse_depth) if sparse_depth is not None else None
        gt_depth = tf_depth_val(gt_depth) if gt_depth is not None else None

        return rgb_dict_tf, sparse_depth, gt_depth

    # In the following, the necessary transforms are specified. Each distinct transform gets an own method for the sake
    # of readability. The idea is to include EVERY transformation (REALLY EVERY TRANSFORMATION!!) applied on the training data (rgb,
    # sparse depth, gt depth) inside of the transformation function
    @staticmethod
    def tf_rgb_train(tgt_size, do_flip, do_aug, aug_params):
        return transforms.Compose(
            [
                tf_prep.PILResize(tgt_size, pil.LANCZOS),
                tf_prep.PILHorizontalFlip(do_flip),
                tf_prep.ColorAug(do_aug, aug_params),
                tf_prep.PrepareForNet()
            ]
        )

    @staticmethod
    def tf_rgb_val(tgt_size):
        return transforms.Compose(
            [
                tf_prep.PILResize(tgt_size, pil.LANCZOS),
                tf_prep.PrepareForNet()
            ]
        )

    @staticmethod
    def tf_depth_train(tgt_size, do_flip):
        return transforms.Compose(
            [
                tf_prep.PILResize(tgt_size, pil.NEAREST),
                tf_prep.PILHorizontalFlip(do_flip),
                tf_prep.TransformToDepthKalimu(),
                tf_prep.PrepareForNet()
            ]
        )

    @staticmethod
    def tf_depth_val(tgt_size):
        return transforms.Compose(
            [
                tf_prep.PILResize(tgt_size, pil.NEAREST),
                tf_prep.TransformToDepthKalimu(),
                tf_prep.PrepareForNet()
            ]
        )

    def get_depth(self, path_file):
        if path_file is None:
            return None
        else:
            assert os.path.exists(path_file), "The file {} does not exist!".format(path_file)
            return pil.open(path_file)

    # Get rgb image or an image that has a frame_id=old_frame_id+offset (to get nearby images)
    def get_rgb(self, path_file, offset):
        assert isinstance(offset, numbers.Number), "The inputted offset {} is not numeric!".format(offset)
        assert os.path.exists(path_file), "The file {} does not exist!".format(path_file)

        tgt_path_file = path_file

        if offset != 0:
            mode, drive_seq, cam_id, frame_id, format_img = decompose_rgb_path(path_file)
            tgt_frame_id = "{:010d}".format(int(frame_id) + offset)
            tgt_path_file = _PathsKalimuFromMotion.get_rgb_image_path(self.paths.path_base, mode, drive_seq, cam_id, tgt_frame_id, format_img)

        if not os.path.exists(tgt_path_file):
            return None

        with open(tgt_path_file, 'rb') as f:
            return pil.open(f).convert('RGB')
import abc
import os
import ntpath
import torch.utils.data as data


class PathsHandlerBase(metaclass=abc.ABCMeta):
    """
    PathsHandler for all datasets with depth and semantic labels. If one of these labels not available please use
    according subclass.
    """
    def __init__(self, mode, split, cfg):
        assert os.path.exists(cfg.dataset.path) == True, 'The entered base path is not valid!'

        self.mode = mode

        # Move parameters from cfg to local variables for readability
        self.use_sparse_depth = cfg.dataset.use_sparse_depth
        if mode is 'train':
            self.gt_available = cfg.eval.train.gt_available
            self.gt_semantic_available = cfg.eval.train.gt_semantic_available
        elif mode is 'val':
            self.gt_available = cfg.eval.val.gt_available
            self.gt_semantic_available = cfg.eval.val.gt_semantic_available
        elif mode is 'test':
            self.gt_available = cfg.eval.test.gt_available
            self.gt_semantic_available = cfg.eval.test.gt_semantic_available
        else:
            assert False, "There is no other mode than 'train', 'val' and 'test'"

        # Get all paths
        self.path_base = cfg.dataset.path
        self.paths_rgb = self.get_rgb_image_paths(mode, split=split)
        self.paths_semantic = self.get_semantic_label_paths(mode, split=split)
        self.paths_depth_sparse = self.get_sparse_depth_image_paths(mode, split=split)
        self.paths_depth_gt = self.get_gt_depth_image_paths(mode, split=split)
        self.path_calib = self.get_calib_path(self.path_base)

        # Adapt the paths for consistency (as there are some depth images missing for some rgb images)
        self.make_consistent()
        assert self.is_consistent() == True

        # Simple consistency check. This is kept here despite the consistency_check method in the deriving class, in
        # case it is kept empty...
        if len(self.paths_rgb) == 0:
            assert False, "No RGB images could be found!"
        if self.gt_available and len(self.paths_depth_gt) == 0:
            assert False, "No dense GT depth images could be found!"
        if self.use_sparse_depth and len(self.paths_depth_sparse) == 0:
            assert False, "No sparse depth images could be found!"
        if self.paths_depth_sparse is not None and len(self.paths_rgb) != len(self.paths_depth_sparse):
            assert False, "There is a mismatch between the number of RGB images ({}) & the number of sparse depth images ({})!".format(len(self.paths_rgb), len(self.paths_depth_sparse))
        if self.paths_depth_gt is not None and len(self.paths_rgb) != len(self.paths_depth_gt):
            assert False, "There is a mismatch between the number of RGB images ({}) & the number of dense GT depth images ({})!".format(len(self.paths_rgb), len(self.paths_depth_gt))

    @abc.abstractmethod
    def get_rgb_image_paths(self):
        pass

    @abc.abstractmethod
    def get_semantic_label_paths(self):
        pass

    @abc.abstractmethod
    def get_gt_depth_image_paths(self):
        pass

    @abc.abstractmethod
    def get_sparse_depth_image_paths(self):
        pass

    @abc.abstractmethod
    def get_calib_path(self):
        pass

    @abc.abstractmethod
    def make_consistent(self):
        pass

    @abc.abstractmethod
    def is_consistent(self):
        pass

    def paths_rgb(self):
        return self.paths_rgb

    def paths_semantic(self):
        return self.paths_semantic

    def paths_depth_sparse(self):
        return self.paths_depth_sparse

    def paths_depth_gt(self):
        return self.paths_depth_gt

    def path_calib(self):
        return self.path_calib

    def mode(self):
        return self.mode()


class PathsHandlerDepth(PathsHandlerBase):
    """
        This class sets all the Not-depth related attributes of the PathsHandlerBase to None.
        This should be used for purely Depth and RGB datasets.
    """
    def __init__(self, mode, split, cfg):
        super(PathsHandlerDepth, self).__init__(mode, split, cfg)

    def get_semantic_label_paths(self, *args, **kwargs):
        return None


class PathsHandlerSemantic(PathsHandlerBase):
    """
        This class sets all the Not-Semantic related attributes of the PathsHandlerBase to None.
        This should be used for purely Semantic and RGB datasets.
    """
    def __init__(self, mode, split, cfg):
        super(PathsHandlerSemantic, self).__init__(mode, split, cfg)

    def get_gt_depth_image_paths(self, *args, **kwargs):
        return None

    def get_sparse_depth_image_paths(self, *args, **kwargs):
        return None

    def get_calib_path(self, *args, **kwargs):
        return None

    def make_consistent(self, *args, **kwargs):
        # datasets without depth grund truth are by definition consistent
        # this should always return True else mistake in implementation of
        # the config file
        if not self.use_sparse_depth and not self.gt_available:
            return True
        else:
            raise('Exception: Semantic Dataset should be consistent, check cfg if cfg.dataset.use_sparse_depth or' \
                  'cfg.eval.<your_train_mode>.gt_available is set to True.')

    def is_consistent(self, *args, **kwargs):
        # datasets without depth grund truth are by definition consistent
        # this should always return True else mistake in implementation of
        # the config file
        if not self.use_sparse_depth and not self.gt_available:
            return True
        else:
            raise('Exception: Semantic Dataset should be consistent, check cfg if cfg.dataset.use_sparse_depth or' \
                  'cfg.eval.<your_train_mode>.gt_available is set to True.')


class DatasetBase(data.Dataset, metaclass=abc.ABCMeta):
    def __init__(self, pathsObj, cfg):
        self.cfg = cfg
        self.paths = pathsObj
        self.mode = pathsObj.mode
        self.rgb_frame_offsets = cfg.train.rgb_frame_offsets
        self.feed_img_size = cfg.dataset.feed_img_size

    @abc.abstractmethod
    def transform_train(self):
        pass

    @abc.abstractmethod
    def transform_val(self):
        pass

    @abc.abstractmethod
    def get_depth(self, path_file):
        pass

    @abc.abstractmethod
    def get_rgb(self, path_file, offset):
        pass

    @abc.abstractmethod
    def get_semantic(self, path_file):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, index):
        pass


class DatasetSemantic(DatasetBase):
    def __init__(self, pathsObj, cfg):
        super(DatasetSemantic, self).__init__(pathsObj, cfg)

    def get_depth(self, path_file):
        return None


class DatasetDepth(DatasetBase):
    def __init__(self, pathsObj, cfg):
        super(DatasetSemantic, self).__init__(pathsObj, cfg)

    def get_semantic(self, path_file):
        return None

    def __len__(self):
        return len(self.paths.paths_rgb)

    def __getitem__(self, index):
        # Get all required training elements
        sparse_depth = self.get_depth(self.paths.paths_depth_sparse[index]) if self.paths.paths_depth_sparse is not None else None

        gt_depth = self.get_depth(self.paths.paths_depth_gt[index]) if self.paths.paths_depth_gt is not None else None

        rgb_imgs = {}
        for offset in self.rgb_frame_offsets:
            if self.get_rgb(self.paths.paths_rgb[index], offset) is not None:
                rgb_imgs[offset] = self.get_rgb(self.paths.paths_rgb[index], offset)
            else: # As we can't train on the first and last image as temporal information is missing, we append the
                # dataset by two images for the beginning and the end by the corresponding non-offset RGB image
                rgb_imgs[offset] = self.get_rgb(self.paths.paths_rgb[index], 0)

        # Attention: GT depth is also resized here
        # Transform the training items (augmentation and preparation for net (i.e. expand dims and ToTensor)
        if self.mode == 'train':
            rgb_imgs, sparse_depth, gt_depth = self.transform_train(rgb_imgs, sparse_depth, gt_depth)
        elif self.mode == 'val' or self.mode =='test':
            rgb_imgs, sparse_depth, gt_depth = self.transform_val(rgb_imgs, sparse_depth, gt_depth)
        else:
            assert False, "The mode {} is not defined!".format(self.mode)

        # Create the map to be outputted
        data = {}
        if sparse_depth is not None:
            data["sparse"] = sparse_depth

        if gt_depth is not None:
            data["gt"] = gt_depth

        for offset, val in rgb_imgs.items():
            data[("rgb", offset)] = val

        return data
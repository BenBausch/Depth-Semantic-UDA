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

        # Get all paths
        self.path_base = cfg.dataset.path
        self.paths_rgb = self.get_rgb_image_paths(mode, split=split)

        if len(self.paths_rgb) == 0:
            assert False, "No RGB images could be found!"

    @abc.abstractmethod
    def get_rgb_image_paths(self):
        pass

    def paths_rgb(self):
        return self.paths_rgb

    def mode(self):
        return self.mode()


class PathsHandlerDepth(PathsHandlerBase):
    """
        This class sets all the Not-depth related attributes of the PathsHandlerRGB to None.
        This should be used for purely Depth and RGB datasets.
    """
    def __init__(self, mode, split, cfg):
        super(PathsHandlerDepth, self).__init__(mode, split, cfg)

        # Move parameters from cfg to local variables for readability
        self.use_sparse_depth = cfg.dataset.use_sparse_depth
        if mode is 'train':
            self.gt_depth_available = cfg.eval.train.gt_depth_available
        elif mode is 'val':
            self.gt_depth_available = cfg.eval.val.gt_depth_available
        elif mode is 'test':
            self.gt_depth_available = cfg.eval.test.gt_depth_available
        else:
            assert False, "There is no other mode than 'train', 'val' and 'test'"

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

    def paths_depth_sparse(self):
        return self.paths_depth_sparse

    def paths_depth_gt(self):
        return self.paths_depth_gt

    def path_calib(self):
        return self.path_calib


class PathsHandlerSemantic(PathsHandlerBase):
    """
        This class sets all the Not-Semantic related attributes of the PathsHandlerRGB to None.
        This should be used for purely Semantic and RGB datasets.
    """
    def __init__(self, mode, split, cfg):
        super(PathsHandlerSemantic, self).__init__(mode, split, cfg)
        # Move parameters from cfg to local variables for readability
        if mode is 'train':
            self.gt_semantic_available = cfg.eval.train.gt_semantic_available
        elif mode is 'val':
            self.gt_semantic_available = cfg.eval.val.gt_semantic_available
        elif mode is 'test':
            self.gt_semantic_available = cfg.eval.test.gt_semantic_available
        else:
            assert False, "There is no other mode than 'train', 'val' and 'test'"

    @abc.abstractmethod
    def get_semantic_label_paths(self):
        pass

    def paths_semantic(self):
        return self.paths_semantic


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
# Reference point for all configurable options regarding an individual dataset like Cityscapes
from yacs.config import CfgNode as CN

cfg_dataset = CN()
cfg_dataset.dataset = CN()

# ********************************************************************
# /----- Dataset parameters
# ********************************************************************
cfg_dataset.dataset.name = ''
cfg_dataset.dataset.path = ''
cfg_dataset.dataset.sub_dataset_paths = []
cfg_dataset.dataset.feed_img_size = []  # used for datasets with fixed image sizes, or when resizing everything to
# same resolution
cfg_dataset.dataset.resize_factor = 1  # used for datasets with variable image sizes
cfg_dataset.dataset.use_sparse_depth = True
cfg_dataset.dataset.use_dense_depth = False
cfg_dataset.dataset.use_self_supervised_depth = True
cfg_dataset.dataset.use_semantic_gt = False
cfg_dataset.dataset.split = None
cfg_dataset.dataset.camera = ''  # 'pinhole'
cfg_dataset.dataset.min_depth = 0.001
cfg_dataset.dataset.max_depth = 80.0
cfg_dataset.dataset.shuffle = False
cfg_dataset.dataset.img_norm = True
cfg_dataset.dataset.do_augmentation = True
cfg_dataset.dataset.num_classes = 1
cfg_dataset.dataset.rgb_frame_offsets = []
cfg_dataset.dataset.sequence_dataset = False
cfg_dataset.dataset.debug = False
cfg_dataset.dataset.predict_depth_for_each_img_in_sequence = False
cfg_dataset.dataset.predict_pseudo_labels = False

# ********************************************************************
# /----- Augmentation parameters
# ********************************************************************
cfg_dataset.dataset.augmentation = CN()
cfg_dataset.dataset.augmentation.brightness_jitter = 0.2
cfg_dataset.dataset.augmentation.contrast_jitter = 0.2
cfg_dataset.dataset.augmentation.saturation_jitter = 0.2
cfg_dataset.dataset.augmentation.hue_jitter = 0.1
cfg_dataset.dataset.augmentation.horizontal_flip = 0.5

# ********************************************************************
# /----- Losses
# ********************************************************************
cfg_dataset.losses = CN()

cfg_dataset.losses.loss_names_and_parameters = None  # loss name is key, value dict of parameters
cfg_dataset.losses.loss_names_and_weights = None  # loss name is key, weight is the value

# ********************************************************************
# /----- Evaluation
# ********************************************************************
cfg_dataset.eval = CN()
cfg_dataset.eval.use_garg_crop = True

cfg_dataset.eval.train = CN()
cfg_dataset.eval.train.gt_depth_available = False
cfg_dataset.eval.train.gt_semantic_available = False
cfg_dataset.eval.train.use_gt_scale = False

cfg_dataset.eval.val = CN()
cfg_dataset.eval.val.gt_depth_available = False
cfg_dataset.eval.val.gt_semantic_available = False
cfg_dataset.eval.val.use_gt_scale = False

cfg_dataset.eval.test = CN()
cfg_dataset.eval.test.gt_depth_available = False
cfg_dataset.eval.test.gt_semantic_available = False
cfg_dataset.eval.test.use_gt_scale = False


def get_cfg_dataset_defaults():
    """Get a yacs CfgNode object with default values"""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg_dataset.clone()

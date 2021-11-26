# Reference point for all configurable options
from yacs.config import CfgNode as CN

# ToDo: Check in the beginning if everything's ok, e.g. whether the specified paths do exist...
# /----- Create a cfg node
cfg = CN()

# ********************************************************************
# /------ Training parameters
# ********************************************************************
cfg.train = CN()
cfg.train.nof_epochs = 20
cfg.train.nof_workers = 4
cfg.train.batch_size = 12
cfg.train.rgb_frame_offsets = [0, -1, +1]

# /----- Optimizer parameters
cfg.train.optimizer = CN()
cfg.train.optimizer.type = 'Adam'
cfg.train.optimizer.learning_rate = 0.0001

# /----- Scheduler parameters
cfg.train.scheduler = CN()
cfg.train.scheduler.type = 'StepLR'
cfg.train.scheduler.step_size = 10
cfg.train.scheduler.gamma = 0.1

# ********************************************************************
# /------ Validation parameters
# ********************************************************************
cfg.val = CN()
cfg.val.batch_size = 12
cfg.val.nof_workers = 4

# ********************************************************************
# /----- Model parameters
# ********************************************************************
cfg.model = CN()

cfg.model.type = ''
cfg.model.depth_net = CN()
cfg.model.depth_net.params = CN()
cfg.model.depth_net.params.nof_layers = 18
cfg.model.depth_net.params.weights_init = 'pretrained'

cfg.model.pose_net = CN()
cfg.model.pose_net.input = 'pairs'
cfg.model.pose_net.params = CN()
cfg.model.pose_net.params.weights_init = 'pretrained'

# ********************************************************************
# /----- Dataset parameters
# ********************************************************************
cfg.dataset = CN()
cfg.dataset.name = '' # 'KITTI'
cfg.dataset.path = '' # '/home/petek/kalimu/data/kitti/base'
cfg.dataset.feed_img_size = [] #[640, 192]
cfg.dataset.use_sparse_depth = True
cfg.dataset.split = None # 'eigen_zhou'
cfg.dataset.camera = '' # 'pinhole'
cfg.dataset.min_depth = 0.001
cfg.dataset.max_depth = 80.0
cfg.dataset.shuffle = False
cfg.dataset.img_norm = False

# ********************************************************************
# /----- Augmentation parameters
# ********************************************************************
cfg.dataset.augmentation = CN()
cfg.dataset.augmentation.brightness_jitter = 0.2
cfg.dataset.augmentation.contrast_jitter = 0.2
cfg.dataset.augmentation.saturation_jitter = 0.2
cfg.dataset.augmentation.hue_jitter = 0.1
cfg.dataset.augmentation.horizontal_flip = 0.5

# ********************************************************************
# /----- Evaluation parameters
# ********************************************************************
cfg.eval = CN()
cfg.eval.use_garg_crop = True

cfg.eval.train = CN()
cfg.eval.train.gt_available = True
cfg.eval.train.gt_semantic_available = False
cfg.eval.train.use_gt_scale = True

cfg.eval.val = CN()
cfg.eval.val.gt_available = True
cfg.eval.val.gt_semantic_available = False
cfg.eval.val.use_gt_scale = True

cfg.eval.test = CN()
cfg.eval.test.gt_available = True
cfg.eval.test.gt_semantic_available = False
cfg.eval.test.use_gt_scale = True

# ********************************************************************
# /----- Losses
# ********************************************************************
cfg.losses = CN()

cfg.losses.use_depth_reprojection_loss = False

cfg.losses.reconstruction = CN()
cfg.losses.reconstruction.nof_scales = 4
cfg.losses.reconstruction.use_ssim = True
cfg.losses.reconstruction.use_automasking = True

cfg.losses.weights = CN()
cfg.losses.weights.depth = 1.0
cfg.losses.weights.smoothness = 0.001
cfg.losses.weights.reconstruction = 1.0

# ********************************************************************
# /----- Device
# ********************************************************************
cfg.device = CN()
cfg.device.no_cuda = False
cfg.device.multiple_gpus = False

# ********************************************************************
# /----- IO
# *******************************************************************
cfg.io = CN()
cfg.io.path_save = '/home/petek/trained_models/depth_estimation'
cfg.io.save_frequency = 1  # number of epochs between each save
cfg.io.log_frequency = 250

# ********************************************************************
# /----- Checkpoint
# *******************************************************************
cfg.checkpoint = CN()
cfg.checkpoint.use_checkpoint = False
cfg.checkpoint.path_base = ''
cfg.checkpoint.filename = ''

# ********************************************************************
# /----- Checkpoint parameters
# ********************************************************************

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values"""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

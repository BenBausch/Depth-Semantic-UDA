# Reference point for all configurable options
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
cfg.train.batch_size = 8
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

cfg.model.type = ''  # which model to use: packnet, monodepth2, ...

cfg.model.encoder = CN()
cfg.model.encoder.params = CN()
cfg.model.encoder.params.nof_layers = 18
cfg.model.encoder.params.weights_init = 'pretrained'

cfg.model.depth_net = CN()
cfg.model.depth_net.params = CN()

cfg.model.semantic_net = CN()
cfg.model.semantic_net.params = CN()

cfg.model.pose_net = CN()
cfg.model.pose_net.input = 'pairs'
cfg.model.pose_net.params = CN()
cfg.model.pose_net.params.nof_layers = cfg.model.encoder.params.nof_layers  # default use same size resnet
# for depth- and posenet
cfg.model.pose_net.params.weights_init = 'pretrained'

# ********************************************************************
# /----- Source Dataset parameters
# ********************************************************************
cfg.source = CN()
cfg.source.name = ''
cfg.source.path = ''
cfg.source.feed_img_size = []
cfg.source.use_sparse_depth = True
cfg.source.use_semantic_gt = True
cfg.source.split = None
cfg.source.camera = ''
cfg.source.min_depth = 0.01
cfg.source.max_depth = 655.35  # these are the default depts of synthia, look up the depth range of your dataset
cfg.source.shuffle = True
cfg.source.img_norm = False
cfg.source.num_classes = 1
cfg.source.sequence_dataset = False # set to true if sequences are available

# ********************************************************************
# /----- Source Dataset Augmentation parameters
# ********************************************************************
cfg.source.augmentation = CN()
cfg.source.augmentation.brightness_jitter = 0.2
cfg.source.augmentation.contrast_jitter = 0.2
cfg.source.augmentation.saturation_jitter = 0.2
cfg.source.augmentation.hue_jitter = 0.1
cfg.source.augmentation.horizontal_flip = 0.5

# ********************************************************************
# /----- Target Dataset parameters
# ********************************************************************
cfg.target = CN()
cfg.target.name = ''
cfg.target.path = ''
cfg.target.feed_img_size = []
cfg.target.use_sparse_depth = False
cfg.target.use_semantic_gt = False
cfg.target.split = None
cfg.target.camera = ''
cfg.target.min_depth = 0.01
cfg.target.max_depth = 655.35  # these are the default depts of synthia, look up the depth range of your dataset
cfg.target.shuffle = False
cfg.target.img_norm = False
cfg.target.num_classes = 1
cfg.target.sequence_dataset = True

# ********************************************************************
# /----- Target Dataset Augmentation parameters
# ********************************************************************
cfg.target.augmentation = CN()
cfg.target.augmentation.brightness_jitter = 0.2
cfg.target.augmentation.contrast_jitter = 0.2
cfg.target.augmentation.saturation_jitter = 0.2
cfg.target.augmentation.hue_jitter = 0.1
cfg.target.augmentation.horizontal_flip = 0.5

# ********************************************************************
# /----- Evaluation parameters
# ********************************************************************
cfg.eval = CN()
cfg.eval.source.use_garg_crop = False

cfg.eval.source.train = CN()
cfg.eval.source.train.gt_depth_available = True
cfg.eval.source.train.gt_semantic_available = True
cfg.eval.source.train.use_gt_scale = True

cfg.eval.source.val = CN()
cfg.eval.source.val.gt_depth_available = True
cfg.eval.source.val.gt_semantic_available = True
cfg.eval.source.val.use_gt_scale = True

cfg.eval.source.test = CN()
cfg.eval.source.test.gt_depth_available = True
cfg.eval.source.test.gt_semantic_available = True
cfg.eval.source.test.use_gt_scale = True

cfg.eval.target.use_garg_crop = False

cfg.eval.target.train = CN()
cfg.eval.target.train.gt_depth_available = False
cfg.eval.target.train.gt_semantic_available = False
cfg.eval.target.train.use_gt_scale = False

cfg.eval.target.val = CN()
cfg.eval.target.val.gt_depth_available = False
cfg.eval.target.val.gt_semantic_available = False
cfg.eval.target.val.use_gt_scale = False

cfg.eval.target.test = CN()
cfg.eval.target.test.gt_depth_available = False
cfg.eval.target.test.gt_semantic_available = False
cfg.eval.target.test.use_gt_scale = False


# ********************************************************************
# /----- Losses
# ********************************************************************
cfg.losses = CN()

cfg.losses.source.loss_names_and_parameters = {}  # loss name is key, value dict of parameters
cfg.losses.source.loss_names_and_weights = {}  # loss name is key, weight is the value

cfg.losses.target.loss_names_and_parameters = {}  # loss name is key, value dict of parameters
cfg.losses.target.loss_names_and_weights = {}  # loss name is key, weight is the value


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

# Reference point for all configurable options
# Reference point for all configurable options
from yacs.config import CfgNode as CN
from cfg.config_dataset import get_cfg_dataset_defaults
from copy import deepcopy

# ToDo: Check in the beginning if everything's ok, e.g. whether the specified paths do exist...
# /----- Create a cfg node
cfg = CN()
cfg.experiment_name = ''

# ********************************************************************
# /------ Training parameters
# ********************************************************************
cfg.train = CN()
cfg.train.nof_epochs = 20
cfg.train.nof_workers = 4
cfg.train.batch_size = 8

# /----- Optimizer parameters
cfg.train.optimizer = CN()
cfg.train.optimizer.type = 'Adam'
cfg.train.optimizer.learning_rate = 0.0001

# /----- Scheduler parameters
cfg.train.scheduler = CN()
cfg.train.scheduler.type = 'StepLR'
cfg.train.scheduler.step_size = 10
cfg.train.scheduler.gamma = 0.1
cfg.train.scheduler.milestones = [] # needed only if scheduler type is MultiStepLR

# ********************************************************************
# /------ Validation parameters
# ********************************************************************
cfg.val = CN()
cfg.val.do_validation = True
cfg.val.batch_size = 12
cfg.val.nof_workers = 4

# ********************************************************************
# /----- Model parameters
# ********************************************************************
cfg.model = CN()

cfg.model.type = ''  # which model to use: guda, monodepth2, ...
cfg.model.create_expo_moving_avg_model_copy = False

cfg.model.encoder = CN()
cfg.model.encoder.params = CN()
cfg.model.encoder.params.nof_layers = 18
cfg.model.encoder.params.weights_init = 'pretrained'

cfg.model.depth_net = CN()
cfg.model.depth_net.params = CN()
cfg.model.depth_net.nof_scales = 4

cfg.model.semantic_net = CN()
cfg.model.semantic_net.params = CN()

cfg.model.pose_net = CN()
cfg.model.pose_net.input = 'pairs'
cfg.model.pose_net.params = CN()
cfg.model.pose_net.params.nof_layers = cfg.model.encoder.params.nof_layers  # default use same size resnet
# for depth- and posenet
cfg.model.pose_net.params.weights_init = 'pretrained'
cfg.model.pose_net.params.predict_motion_map = False

# ********************************************************************
# /----- Datasets
# ********************************************************************
cfg.datasets = CN()
cfg.datasets.paths_to_configs = []  # to be set in the yaml file of training,
# paths_to_configs format: single path, or source then target path
cfg.datasets.configs = []  # dataset configs will be initialized in get configs
cfg.datasets.loss_weights = []  # should be same length as cfg.datasets.paths_to_configs

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
cfg.io.path_save = ''
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


def create_configuration(path_to_training_yaml):
    """
    Creates a configuration object.
    :param path_to_training_yaml: path to the yaml configuration file
    :return: configuration object
    """
    configuration = get_cfg_defaults()
    configuration.merge_from_file(path_to_training_yaml)
    if len(configuration.datasets.paths_to_configs) == 0:
        raise Warning('No dataset specified!')

    # assure one loss weight per dataset
    assert len(configuration.datasets.loss_weights) == len(configuration.datasets.paths_to_configs)

    for dataset_configuration in configuration.datasets.paths_to_configs:
        dataset_config = get_cfg_dataset_defaults()
        dataset_config.merge_from_file(dataset_configuration)

        # process losses to be list of dictionary instead of list of multiple dictionaries for easier access without
        # slicing
        loss_dictionary = {}
        if dataset_config.losses.loss_names_and_parameters is not None:
            for loss in dataset_config.losses.loss_names_and_parameters:
                for k in loss.keys():
                    loss_dictionary[k] = loss[k]
            dataset_config.losses.loss_names_and_parameters = [loss_dictionary]

        loss_weight_dictionary = {}
        if dataset_config.losses.loss_names_and_weights is not None:
            for loss in dataset_config.losses.loss_names_and_weights:
                for k in loss.keys():
                    loss_weight_dictionary[k] = loss[k]
            dataset_config.losses.loss_names_and_weights = [loss_weight_dictionary]

        dataset_config.freeze()

        configuration.datasets.configs.append(dataset_config)

    if len(configuration.datasets.configs) == 1:
        print(f'Single dataset specified: {configuration.datasets.configs[0].dataset.name}')

    if len(configuration.datasets.configs) == 2:
        print(f'Two datasets specified: source is {configuration.datasets.configs[0].dataset.name} and '
              f'target is {configuration.datasets.configs[1].dataset.name}')

    if len(configuration.datasets.configs) == 3:
        print(f'Three datasets specified: source is {configuration.datasets.configs[0].dataset.name} and '
              f'target is {configuration.datasets.configs[1].dataset.name} and '
              f'validation dataset is {configuration.datasets.configs[2].dataset.name}')

    configuration.freeze()
    return configuration


def to_dictionary(value):
    """
    Creates a dictionary copy of the yacs configuration.
    :param value: can either be a yacs cfgnode (configuration) or any other object
    :return:
    """
    value = deepcopy(value)  # prevent changing the underlying objects
    if isinstance(value, CN):
        # if value is a cfgNode unpack it into a dictionary
        value = dict(value)
        for key, val in value.items():
            # repeat the same for all sub cfgNodes
            if isinstance(val, list):
                # check if objects in lists are not cfgNodes
                for i, list_val in enumerate(val):
                    val[i] = to_dictionary(list_val)
            else:
                value[key] = to_dictionary(val)
        return value
    else:
        return value


if __name__ == '__main__':
    configuration_path = r'C:\Users\benba\Documents\University\Masterarbeit\Depth-Semantic-UDA\cfg\yaml_files\train' \
                         r'\guda\train_guda_synthia_cityscapes.yaml'
    configi = create_configuration(configuration_path)

    configi_dict = to_dictionary(configi)

    print(configi_dict)

"""
Create an abstract class to force all subclasses to define the methods provided in the base class
"""
import os
import abc

# Own classes
import models
import dataloaders
from io_utils import io_utils

from datetime import datetime
from utils.utils import info_gpu_memory
# PyTorch
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter


# TODO: You might not be able to normalize any camera model. So maybe we shouldn't assume that the camera intrinsics
#  are normalized...
class TrainBase(metaclass=abc.ABCMeta):
    def __init__(self, cfg):
        self.cfg = cfg

        # Specify the device to perform computations on
        if not self.cfg.device.no_cuda and not torch.cuda.is_available():
            raise Exception("There is no GPU available for usage.")

        self.device = torch.device("cuda" if not self.cfg.device.no_cuda else "cpu")
        torch.backends.cudnn.benchmark = True if not self.cfg.device.no_cuda else False

        print("Using '{}' for computation.".format(self.device))
        print("Using cudnn.benchmark? '{}'".format(torch.backends.cudnn.benchmark))

        # Initialize models and send to Cuda if possible
        self.model = models.get_model(self.cfg.model.type, self.device, self.cfg)

        # Initialization of the optimizer and lr scheduler
        self.optimizer = self.get_optimizer(self.cfg.train.optimizer.type,
                                            self.model.parameters_to_train,
                                            self.cfg.train.optimizer.learning_rate)
        self.scheduler = self.get_lr_scheduler(self.optimizer, self.cfg)

        # Get training and validation datasets
        self.train_loader = self.get_dataloader("train")
        self.val_loader = self.get_dataloader("val")

        # Set up IO handler
        path_save_folder = os.path.join(self.cfg.io.path_save, "tmp")
        self.io_handler = io_utils.IOHandler(path_save_folder)

        # get todays date as string 'YYYY_MM_DD' to indicate when the training started
        current_time = datetime.now()
        self.training_start_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")

        # Set up SummaryWriters to log data into the tensorboard events file
        self.writer_train = SummaryWriter(os.path.join(path_save_folder, \
                                                       self.training_start_time + "_" + self.cfg.dataset.name, \
                                                       "tensorboard", "train"))
        self.writer_val = SummaryWriter(os.path.join(path_save_folder, \
                                                       self.training_start_time + "_" + self.cfg.dataset.name, \
                                                       "tensorboard", "val"))
        self.min_depth = self.cfg.dataset.min_depth
        self.max_depth = self.cfg.dataset.max_depth

        # Get number of total steps to compute remaining training time later
        self.num_total_steps = self.num_train_files // self.cfg.train.batch_size * self.cfg.train.nof_epochs

        # Save the current configuration of arguments
        self.io_handler.save_cfg({"time": self.training_start_time, "dataset": self.cfg.dataset.name}, self.cfg)

        # See whether a checkpoint shall be used and load the corresponding weights

        # Load the checkpoint if one is provided
        if cfg.checkpoint.use_checkpoint:
            print("Using pretrained weights for the model and optimizer from \n", os.path.join(cfg.checkpoint.path_base, cfg.checkpoint.filename))
            checkpoint = io_utils.IOHandler.load_checkpoint(cfg.checkpoint.path_base, cfg.checkpoint.filename)
            # Load pretrained weights for the model and the optimizer status
            io_utils.IOHandler.load_weights(checkpoint, self.model.get_networks(), self.optimizer)
        else:
            print("No checkpoint is used. Training from scratch!")

        info_gpu_memory()

    @abc.abstractmethod
    def train(self):
        pass

    def save_checkpoint(self):
        checkpoint = io_utils.IOHandler.gen_checkpoint(
            self.model.get_networks(),
            **{"optimizer": self.optimizer.state_dict(),
                "cfg": self.cfg})

        self.io_handler.save_checkpoint(
            {"epoch": self.epoch, "time": self.training_start_time, "dataset": self.cfg.dataset.name}, checkpoint)

    def get_dataloader(self, mode):
        dataset = dataloaders.get_dataset(self.cfg.dataset.name, mode, self.cfg.dataset.split, self.cfg)

        if mode == "train":
            self.num_train_files = dataset.__len__()
            return DataLoader(dataset, batch_size=self.cfg.train.batch_size, shuffle=True,
                              num_workers=self.cfg.train.nof_workers, pin_memory=True, drop_last=False)

        return DataLoader(dataset, batch_size=self.cfg.val.batch_size, shuffle=False,
                              num_workers=self.cfg.val.nof_workers, pin_memory=True, drop_last=False)

    # Define those methods that can be used for any kind of depth estimation algorithms
    def get_lr_scheduler(self, optimizer, cfg):
        assert isinstance(cfg.train.scheduler.type, str), "The option cfg.train.scheduler.type has to be a string. Current type: {}".format(cfg.train.scheduler.type.type())
        # You can implement other rules here...
        if cfg.train.scheduler.type == "StepLR":
            return lr_scheduler.StepLR(optimizer, step_size=cfg.train.scheduler.step_size, gamma=cfg.train.scheduler.gamma)
        elif cfg.train.scheduler.type == "None":
            return None
        else:
            raise NotImplementedError("The lr scheduler ({}) is not yet implemented.".format(cfg.train.scheduler.type))


    def get_optimizer(self, type_optimizer, params_to_train, learning_rate):
        assert isinstance(type_optimizer, str)
        # You can implement other rules here...
        if type_optimizer == "Adam":
            return torch.optim.Adam(params_to_train, lr=learning_rate)
        elif type_optimizer == "None":
            return None
        else:
            raise NotImplementedError(
                "The optimizer ({}) is not yet implemented.".format(type_optimizer))

"""
Create an abstract class to force all subclasses to define the methods provided in the base class
"""
import os
import abc

# Own classes
from abc import ABC

from cfg.config_training import to_dictionary

import models
import dataloaders
from dataloaders.datasamplers.CompletlyRandomSampler import CompletelyRandomSampler
from io_utils import io_utils

from datetime import datetime
from utils.utils import info_gpu_memory
# PyTorch
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import wandb


class TrainBase(metaclass=abc.ABCMeta):
    def __init__(self, cfg, dataset_type):
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

        # Set up IO handler
        self.path_save_folder = os.path.join(self.cfg.io.path_save, "tmp")
        self.io_handler = io_utils.IOHandler(self.path_save_folder)

        # get todays date as string 'YYYY_MM_DD' to indicate when the training started
        current_time = datetime.now()
        self.training_start_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")

        # See whether a checkpoint shall be used and load the corresponding weights
        # Load the checkpoint if one is provided
        if cfg.checkpoint.use_checkpoint:
            print("Using pretrained weights for the model and optimizer from \n",
                  os.path.join(cfg.checkpoint.path_base, cfg.checkpoint.filename))
            checkpoint = io_utils.IOHandler.load_checkpoint(cfg.checkpoint.path_base, cfg.checkpoint.filename)
            # Load pretrained weights for the model and the optimizer status
            io_utils.IOHandler.load_weights(checkpoint, self.model.get_networks(), self.optimizer)
        else:
            print("No checkpoint is used. Training from scratch!")

        # Logging
        wandb.init(project=f"{self.cfg.experiment_name}", config=to_dictionary(self.cfg))


    @abc.abstractmethod
    def train(self):
        pass

    def save_checkpoint(self):
        checkpoint = io_utils.IOHandler.gen_checkpoint(
            self.model.get_networks(),
            **{"optimizer": self.optimizer.state_dict(),
               "cfg": self.cfg})

        self.io_handler.save_checkpoint(
            {"epoch": self.epoch, "time": self.training_start_time, "dataset": self.cfg.datasets.configs[0].dataset.name},
            checkpoint)

    @staticmethod
    def get_dataloader(mode, name, split, bs, num_workers, cfg, sample_completely_random=False, num_samples=None):
        """
        Creates a dataloader.
        :param cfg: configuration of the dataset (not the whole configuration of training!)
        :param mode: 'train', 'test', 'val'
        :param name: name of the dataset (list of names --> see init dataloaders module)
        :param split: wanted split, e.g. eigen split for kitti_depricated
        :param bs: batch size of the dataloader
        :param num_workers: number of workers for the dataloader
        :param sample_completely_random: randomly sample batches from whole dataset even across epochs
        :param num_samples: length of the dataloader, e.g. number of samples to sample
        :return: dataloader, dataloader length
        """
        dataset = dataloaders.get_dataset(name, mode, split, cfg)

        if sample_completely_random:
            if num_samples is None:
                num_samples = len(dataset)
                raise Warning('num_samples is set to None, if wanted please ignore this message!')
            if mode == "train":
                return DataLoader(dataset, batch_size=bs, num_workers=num_workers,
                                  pin_memory=True, drop_last=False,
                                  sampler=CompletelyRandomSampler(data_source=dataset, num_samples=num_samples)), \
                        num_samples
            else:
                return DataLoader(dataset, batch_size=bs,
                                  num_workers=num_workers, pin_memory=True, drop_last=False,
                                  sampler=CompletelyRandomSampler(data_source=dataset, num_samples=num_samples)), \
                        num_samples
        else:
            if mode == "train":
                return DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers,
                                  pin_memory=True, drop_last=False), len(dataset)
            else:
                return DataLoader(dataset, batch_size=bs, shuffle=False,
                                  num_workers=num_workers, pin_memory=True, drop_last=False), len(dataset)

    # Define those methods that can be used for any kind of depth estimation algorithms
    @staticmethod
    def get_lr_scheduler(optimizer, cfg):
        assert isinstance(cfg.train.scheduler.type,
                          str), "The option cfg.train.scheduler.type has to be a string. Current type: {}".format(
            cfg.train.scheduler.type.type())
        # You can implement other rules here...
        if cfg.train.scheduler.type == "StepLR":
            return lr_scheduler.StepLR(optimizer, step_size=cfg.train.scheduler.step_size,
                                       gamma=cfg.train.scheduler.gamma)
        elif cfg.train.scheduler.type == "MultiStepLR":
            return lr_scheduler.MultiStepLR(optimizer, milestones=cfg.train.scheduler.milestones,
                                            gamma=cfg.train.scheduler.gamma)

        elif cfg.train.scheduler.type == "None":
            return None
        else:
            raise NotImplementedError("The lr scheduler ({}) is not yet implemented.".format(cfg.train.scheduler.type))

    @staticmethod
    def get_optimizer(type_optimizer, params_to_train, learning_rate):
        assert isinstance(type_optimizer, str)
        # You can implement other rules here...
        if type_optimizer == "Adam":
            return torch.optim.Adam(params_to_train, lr=learning_rate)
        elif type_optimizer == "None":
            return None
        else:
            raise NotImplementedError(
                "The optimizer ({}) is not yet implemented.".format(type_optimizer))


# TODO: You might not be able to normalize any camera model. So maybe we shouldn't assume that the camera intrinsics
#  are normalized...
class TrainSingleDatasetBase(TrainBase, ABC):
    def __init__(self, cfg, dataset_type='single'):
        super(TrainSingleDatasetBase, self).__init__(cfg=cfg, dataset_type=dataset_type)

        # Get training and validation datasets
        self.train_loader, self.num_train_files = self.get_dataloader(mode="train",
                                                                      name=self.cfg.datasets.configs[0].dataset.name,
                                                                      split=self.cfg.datasets.configs[0].dataset.split,
                                                                      bs=self.cfg.train.batch_size,
                                                                      num_workers=self.cfg.train.nof_workers,
                                                                      cfg=self.cfg.datasets.configs[0])
        print(f'Length Train Loader: {len(self.train_loader)}')
        self.val_loader, self.num_val_files = self.get_dataloader(mode="val",
                                                                  name=self.cfg.datasets.configs[0].dataset.name,
                                                                  split=self.cfg.datasets.configs[0].dataset.split,
                                                                  bs=self.cfg.val.batch_size,
                                                                  num_workers=self.cfg.val.nof_workers,
                                                                  cfg=self.cfg.datasets.configs[0])
        print(f'Length Validation Loader: {len(self.val_loader)}')

        # Get number of total steps to compute remaining training time later
        self.num_total_steps = self.num_train_files // self.cfg.train.batch_size * self.cfg.train.nof_epochs

        # Save the current configuration of arguments
        self.io_handler.save_cfg({"time": self.training_start_time, "dataset": self.cfg.datasets.configs[0].dataset.name},
                                 self.cfg)

        # Load the checkpoint if one is provided
        if cfg.checkpoint.use_checkpoint:
            print("Using pretrained weights for the model and optimizer from \n",
                  os.path.join(cfg.checkpoint.path_base, cfg.checkpoint.filename))
            checkpoint = io_utils.IOHandler.load_checkpoint(cfg.checkpoint.path_base, cfg.checkpoint.filename)
            # Load pretrained weights for the model and the optimizer status
            io_utils.IOHandler.load_weights(checkpoint, self.model.get_networks(), self.optimizer)
        else:
            print("No checkpoint is used. Training from scratch!")


class TrainSourceTargetDatasetBase(TrainBase, ABC):
    def __init__(self, cfg, dataset_type='target'):
        super(TrainSourceTargetDatasetBase, self).__init__(cfg=cfg, dataset_type=dataset_type)

        # Get training and validation datasets for source and target
        self.target_train_loader, self.target_num_train_files = \
            self.get_dataloader(mode="train",
                                name=self.cfg.datasets.configs[1].dataset.name,
                                split=self.cfg.datasets.configs[1].dataset.split,
                                bs=self.cfg.train.batch_size,
                                num_workers=self.cfg.train.nof_workers,
                                cfg=self.cfg.datasets.configs[1])
        print(f'Length Target Train Loader: {len(self.target_train_loader)}')
        self.target_val_loader, self.target_num_val_files = \
            self.get_dataloader(mode="val",
                                name=self.cfg.datasets.configs[1].dataset.name,
                                split=self.cfg.datasets.configs[1].dataset.split,
                                bs=self.cfg.val.batch_size,
                                num_workers=self.cfg.val.nof_workers,
                                cfg=self.cfg.datasets.configs[1])
        print(f'Length Target Validation Loader: {len(self.target_val_loader)}')

        self.source_train_loader, self.source_num_train_files = \
            self.get_dataloader(mode="train",
                                name=self.cfg.datasets.configs[0].dataset.name,
                                split=self.cfg.datasets.configs[0].dataset.split,
                                bs=self.cfg.train.batch_size,
                                num_workers=self.cfg.train.nof_workers,
                                cfg=self.cfg.datasets.configs[0],
                                sample_completely_random=True,
                                num_samples=self.target_num_train_files)
        print(f'Length Source Train Loader: {len(self.source_train_loader)}')
        self.source_val_loader, self.source_num_val_files = \
            self.get_dataloader(mode="val",
                                name=self.cfg.datasets.configs[0].dataset.name,
                                split=self.cfg.datasets.configs[0].dataset.split,
                                bs=self.cfg.val.batch_size,
                                num_workers=self.cfg.val.nof_workers,
                                cfg=self.cfg.datasets.configs[0])
        print(f'Length Source Validation Loader: {len(self.source_val_loader)}')

        # Get number of total steps to compute remaining training time later
        # calculate time using target dataset length
        self.num_total_steps = self.target_num_train_files // self.cfg.train.batch_size * self.cfg.train.nof_epochs

        # Save the current configuration of arguments
        self.io_handler.save_cfg({"time": self.training_start_time, "dataset": self.cfg.datasets.configs[0].dataset.name
                                  + "_" + self.cfg.datasets.configs[1].dataset.name}, self.cfg)

        # Load the checkpoint if one is provided
        if cfg.checkpoint.use_checkpoint:
            print("Using pretrained weights for the model and optimizer from \n",
                  os.path.join(cfg.checkpoint.path_base, cfg.checkpoint.filename))
            checkpoint = io_utils.IOHandler.load_checkpoint(cfg.checkpoint.path_base, cfg.checkpoint.filename)
            # Load pretrained weights for the model and the optimizer status
            io_utils.IOHandler.load_weights(checkpoint, self.model.get_networks(), self.optimizer)
        else:
            print("No checkpoint is used. Training from scratch!")

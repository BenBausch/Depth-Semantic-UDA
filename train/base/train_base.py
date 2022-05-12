"""
Create an abstract class to force all subclasses to define the methods provided in the base class
"""
import os
import abc
import random
from abc import ABC
import re

# Own classes
import numpy.random
import numpy as np
from utils.plotting_like_cityscapes_utils import semantic_id_tensor_to_rgb_numpy_array as s_to_rgb
from utils.plotting_like_cityscapes_utils import visu_depth_prediction as vdp
from cfg.config_training import to_dictionary
import models
import dataloaders
from io_utils import io_utils
from datetime import datetime
from dataloaders.datasamplers import get_sampler

# PyTorch
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from models.helper_models.custom_data_parallel import CustomDistributedDataParallel
import torch.distributed as dist
import wandb

# imports only available on Linux
try:
    from torch.distributed.optim import DistributedOptimizer
except ImportError:
    pass


def setup_multi_processing():
    """
    Set environment variables for ddp training.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12344'


def run_trainer(device_id, cfg, world_size, TrainerClass):
    """Wrapper function to initiate the trainer"""
    trainer = TrainerClass(device_id, cfg, world_size)
    trainer.run()


class TrainBase(metaclass=abc.ABCMeta):
    """Base class for trainers"""
    def __init__(self, device_id, cfg, world_size=1):
        """Loads the ddp or conventional model, datasets, optimizers, learning rate schedulers, ..."""
        self.cfg = cfg

        self.rank = device_id  # shall be defined also for 1 gpu training for less logical branching while training
        if self.cfg.device.multiple_gpus:
            #  setup Multi_device
            self.world_size = world_size
            torch.cuda.set_device(device_id)

            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=self.rank
            )

        # Specify the device to perform computations on
        if not self.cfg.device.no_cuda and not torch.cuda.is_available():
            raise Exception("There is no GPU available for usage.")

        self.device = f"cuda:{device_id}"
        torch.backends.cudnn.benchmark = True if not self.cfg.device.no_cuda else False

        print("Using '{}' for computation.".format(self.device))
        print("Using cudnn.benchmark? '{}'".format(torch.backends.cudnn.benchmark))

        # Initialize models and send to Cuda if possible
        self.model = models.get_model(self.cfg.model.type, self.cfg)
        if self.cfg.device.multiple_gpus:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model.to(self.device)

        if self.cfg.device.multiple_gpus:
            self.model = CustomDistributedDataParallel(self.model, device_ids=[device_id], find_unused_parameters=True)
            torch.manual_seed(self.rank)
            numpy.random.seed(self.rank)
            random.seed(self.rank)

        self.optimizer = self.get_optimizer(self.cfg.train.optimizer.type,
                                            self.model.params_to_train(self.cfg.train.optimizer.learning_rate),
                                            self.cfg.train.optimizer.learning_rate)

        # Initialization of the optimizer and lr scheduler
        self.scheduler = self.get_lr_scheduler(self.optimizer, self.cfg)

        # Set up IO handler
        self.path_save_folder = os.path.join(self.cfg.io.path_save, "tmp")
        self.io_handler = io_utils.IOHandler(self.path_save_folder)

        # get todays date as string 'YYYY_MM_DD' to indicate when the training started
        current_time = datetime.now()
        self.training_start_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")

        # See whether a checkpoint shall be used and load the corresponding weights
        # Load the checkpoint if one is provided
        print(f'Using a checkpoint {cfg.checkpoint.use_checkpoint}')
        if cfg.checkpoint.use_checkpoint:
            print("Using pretrained weights for the model and optimizer from \n",
                  os.path.join(cfg.checkpoint.path_base, cfg.checkpoint.filename))
            checkpoint = io_utils.IOHandler.load_checkpoint(cfg.checkpoint.path_base, cfg.checkpoint.filename, self.rank)
            # Load pretrained weights for the model and the optimizer status
            io_utils.IOHandler.load_weights(checkpoint, self.model.get_networks(), self.optimizer)
        else:
            print("No checkpoint is used. Training from scratch!")

        # Logging only on the first gpu process
        if device_id == 0:
            print("Training will use ", torch.cuda.device_count(), "GPUs!")
            wandb.init(project=f"{self.cfg.experiment_name}", config=to_dictionary(self.cfg))

    @abc.abstractmethod
    def train(self):
        """Main training function to be implemented."""
        pass

    def save_checkpoint(self):
        """Creates and saves a checkpoint of the model and the information necessary."""
        checkpoint = io_utils.IOHandler.gen_checkpoint(
            self.model.get_networks(),
            **{"optimizer": self.optimizer.state_dict(),
               "cfg": self.cfg})

        self.io_handler.save_checkpoint(
            {"epoch": self.epoch, "time": self.training_start_time,
             "dataset": self.cfg.datasets.configs[0].dataset.name},
            checkpoint)

    def get_dataloader(self, mode, name, split, bs, num_workers, cfg, sample_completely_random=False, num_samples=None):
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
        if mode == 'train':
            shuffle = True

            if sample_completely_random:
                # samples are completely random at each epoch not like pytorch's random sampler.
                # no need for distributed version of completely random sampler, since sampling with replacement does not
                # require considering sampled ids of other gpus.
                if num_samples is None:
                    num_samples = len(dataset)
                sampler = get_sampler("CompletelyRandomSampler",
                                      data_source=dataset,
                                      num_samples=num_samples)
            else:
                # samples should be sampled sequentially, either standard fashion or in distributed data parallel scheme
                if self.cfg.device.multiple_gpus:
                    # distributed data parallel scheme
                    sampler = get_sampler("DistributedSampler",
                                          dataset,
                                          num_replicas=self.world_size,
                                          rank=self.rank,
                                          shuffle=shuffle)
                else:
                    # standard fashion
                    sampler = get_sampler("RandomSampler", dataset)

        else:
            # always validate on single gpu (easiest ways to calculate validation metrics,
            # like miou calculated over whole validation set, even if training done in
            # multi-gpu setting, no gathering of values across gpus required)
            sampler = get_sampler("SequentialSampler", dataset)

        loader = DataLoader(dataset,
                            batch_size=bs,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False,
                            sampler=sampler)

        return loader, len(loader)

    def print_p_0(self, obj):
        """
        Prints string of object only if the process has id 0.
        Example use case: print loss during training
        :param object:
        :return:
        """
        if self.rank == 0:
            print(str(obj))

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
            optimizer = torch.optim.Adam(params_to_train, lr=learning_rate)
            return optimizer
        elif type_optimizer == "None":
            return None
        else:
            raise NotImplementedError(
                "The optimizer ({}) is not yet implemented.".format(type_optimizer))

    def set_train(self):
        for m in self.model.networks.values():
            m.train()

    def set_eval(self):
        for m in self.model.networks.values():
            m.eval()

    def set_from_checkpoint(self):
        """
        Set the parameters to the parameters of the safed checkpoints.
        """
        # get epoch from file name
        self.epoch = int(re.match("checkpoint_epoch_([0-9]*).pth", self.cfg.checkpoint.filename).group(1)) + 1

    def get_wandb_depth_image(self, depth, batch_idx, caption_addon=''):
        """
        Creates a Wandb depth image with yellow (close) to far (black) encoding
        :param depth: depth batch, prediction or ground truth
        :param batch_idx: idx of batch in the epoch
        :param caption_addon: string
        """
        colormapped_depth = vdp(1 / depth)
        img = wandb.Image(colormapped_depth, caption=f'Depth Map of {batch_idx}' + caption_addon)
        return img

    def get_wandb_normal_image(self, depth, snr_module, caption_addon=''):
        """
        Creates a Wandb image of the surface normals with coordinate to rgb encoding x-->R, y-->G, z-->B
        """
        points3d = snr_module.img_to_pointcloud(depth)
        normals = snr_module.get_normal_vectors(points3d)

        normals_plot = normals * 255

        normals_plot = normals_plot[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

        return wandb.Image(normals_plot, caption=f'{caption_addon} Normal')

    def get_wandb_semantic_image(self, semantic, is_prediction=True, k=1, caption=''):
        """
        Creates a Wandb image of the semantic segmentation in cityscapes rgb encoding.
        :param semantic: semantic sample
        :param is_prediction: true if sample is a prediction
        :param k: top k prediction to be plotted
        :param caption: string
        """
        if is_prediction:
            semantic = torch.topk(semantic, k=k, dim=0, sorted=True).indices[k - 1].unsqueeze(0)
        else:
            semantic = semantic.unsqueeze(0)
        img = wandb.Image(s_to_rgb(semantic.detach().cpu(), num_classes=self.num_classes), caption=caption)
        return img


class TrainSingleDatasetBase(TrainBase, ABC):
    """Used for training on a single dataset and validation on a second dataset."""
    def __init__(self, device_id, cfg, world_size=1):
        super(TrainSingleDatasetBase, self).__init__(cfg=cfg, device_id=device_id, world_size=world_size)

        # Get training and validation datasets
        self.train_loader, self.num_train_files = self.get_dataloader(mode="train",
                                                                      name=self.cfg.datasets.configs[0].dataset.name,
                                                                      split=self.cfg.datasets.configs[0].dataset.split,
                                                                      bs=self.cfg.train.batch_size,
                                                                      num_workers=self.cfg.train.nof_workers,
                                                                      cfg=self.cfg.datasets.configs[0])
        print(f'Length Train Loader: {len(self.train_loader)}')
        self.val_loader, self.num_val_files = self.get_dataloader(mode="val",
                                                                  name=self.cfg.datasets.configs[1].dataset.name,
                                                                  split=self.cfg.datasets.configs[1].dataset.split,
                                                                  bs=self.cfg.val.batch_size,
                                                                  num_workers=self.cfg.val.nof_workers,
                                                                  cfg=self.cfg.datasets.configs[1])
        print(f'Length Validation Loader: {len(self.val_loader)}')

        # Get number of total steps to compute remaining training time later
        self.num_total_steps = self.num_train_files // self.cfg.train.batch_size * self.cfg.train.nof_epochs


class TrainSourceTargetDatasetBase(TrainBase, ABC):
    """Used for training on two datasets (each sample gets an equal length) and validation on a second dataset."""
    def __init__(self, device_id, cfg, world_size=1):
        super(TrainSourceTargetDatasetBase, self).__init__(cfg=cfg, device_id=device_id, world_size=world_size)

        num_files_per_epoch = 9400

        # Get training and validation datasets for source and target
        self.target_train_loader, self.target_num_train_files = \
            self.get_dataloader(mode="train",
                                name=self.cfg.datasets.configs[1].dataset.name,
                                split=self.cfg.datasets.configs[1].dataset.split,
                                bs=self.cfg.train.batch_size,
                                num_workers=self.cfg.train.nof_workers,
                                cfg=self.cfg.datasets.configs[1],
                                sample_completely_random=True,
                                num_samples=int(num_files_per_epoch/torch.cuda.device_count()))
        print(f'Length Target Train Loader: {len(self.target_train_loader)}')

        self.target_val_loader, self.target_num_val_files = \
            self.get_dataloader(mode="val",
                                name=self.cfg.datasets.configs[2].dataset.name,
                                split=self.cfg.datasets.configs[2].dataset.split,
                                bs=self.cfg.val.batch_size,
                                num_workers=self.cfg.val.nof_workers,
                                cfg=self.cfg.datasets.configs[2])
        print(f'Length Target Validation Loader: {len(self.target_val_loader)}')

        self.source_train_loader, self.source_num_train_files = \
            self.get_dataloader(mode="train",
                                name=self.cfg.datasets.configs[0].dataset.name,
                                split=self.cfg.datasets.configs[0].dataset.split,
                                bs=self.cfg.train.batch_size,
                                num_workers=self.cfg.train.nof_workers,
                                cfg=self.cfg.datasets.configs[0],
                                sample_completely_random=True,
                                num_samples=int(num_files_per_epoch/torch.cuda.device_count()))
        print(f'Length Source Train Loader: {len(self.source_train_loader)}')

        # 4th dataset considered the Augmented dataset being trained with gt and pseudo_labels
        self.use_mixed_dataset = False
        if len(self.cfg.datasets.configs) == 4:
            self.use_mixed_dataset = True
            print(f'Using 4 datasets, the last dataset is the augmented dataset!')
            self.mixed_train_loader, self.mixed_num_train_files = \
                self.get_dataloader(mode="train",
                                    name=self.cfg.datasets.configs[3].dataset.name,
                                    split=self.cfg.datasets.configs[3].dataset.split,
                                    bs=self.cfg.train.batch_size,
                                    num_workers=self.cfg.train.nof_workers,
                                    cfg=self.cfg.datasets.configs[3],
                                    sample_completely_random=True,
                                    num_samples=int(num_files_per_epoch/torch.cuda.device_count()))

        # Get number of total steps to compute remaining training time later
        # calculate time using target dataset length
        self.num_total_steps = self.target_num_train_files // self.cfg.train.batch_size * self.cfg.train.nof_epochs


# Python modules
import time
import os
import re

import numpy as np
import torch
import matplotlib.pyplot as plt

# Own classes
from helper_modules.image_warper import _ImageToPointcloud
from utils.plotting_like_cityscapes_utils import semantic_id_tensor_to_rgb_numpy_array as s_to_rgb
from utils.plotting_like_cityscapes_utils import visu_depth_prediction as vdp
from losses.metrics import MIoU, DepthEvaluator
from train.base.train_base import TrainSingleDatasetBase
from losses import get_loss
import camera_models
import wandb
import torch.nn.functional as F
from utils.plotting_like_cityscapes_utils import CITYSCAPES_ID_TO_NAME_19, CITYSCAPES_ID_TO_NAME_16
from utils.constans import IGNORE_VALUE_DEPTH, IGNORE_INDEX_SEMANTIC
from utils import get_cross_entropy_weights


class Depth2SemanticTrainer(TrainSingleDatasetBase):
    """
    Training on datasets with semantic labels, like Synthia or Cityscapes Semantic dataset
    """
    def __init__(self, device_id, cfg, world_size=1):
        super(Depth2SemanticTrainer, self).__init__(device_id=device_id, cfg=cfg, world_size=world_size)

        # -------------------------Parameters that should be the same for all datasets----------------
        self.num_classes = self.cfg.datasets.configs[0].dataset.num_classes
        for i, dcfg in enumerate(self.cfg.datasets.configs):
            assert self.num_classes == dcfg.dataset.num_classes

        if self.num_classes == 19:
            self.c_id_to_name = CITYSCAPES_ID_TO_NAME_19
        elif self.num_classes == 16:
            self.c_id_to_name = CITYSCAPES_ID_TO_NAME_16
        else:
            raise ValueError(f"GUDA training not defined for {self.num_classes} classes!")

        # -------------------------Source dataset parameters------------------------------------------
        self.img_width = self.cfg.datasets.configs[0].dataset.feed_img_size[0]
        self.img_height = self.cfg.datasets.configs[0].dataset.feed_img_size[1]
        assert self.img_height % 32 == 0, 'The image height must be a multiple of 32'
        assert self.img_width % 32 == 0, 'The image width must be a multiple of 32'

        l_n_p = self.cfg.datasets.configs[0].losses.loss_names_and_parameters
        l_n_w = self.cfg.datasets.configs[0].losses.loss_names_and_weights

        # Specify whether to train in fully unsupervised manner or not
        if self.cfg.datasets.configs[0].dataset.use_semantic_gt:
            self.print_p_0('Training supervised on source dataset using semantic annotations!')

        # -------------------------Image to point cloud stuff ----------------------------------------

        # Set up normalized camera model for source domain
        try:
            self.train_camera_model = \
                camera_models.get_camera_model_from_file(self.cfg.datasets.configs[0].dataset.camera,
                                                         os.path.join(self.cfg.datasets.configs[0].dataset.path,
                                                                      "calib", "calib.txt"))

        except FileNotFoundError:
            raise Exception("No Camera calibration file found! Create Camera calibration file under: " +
                            os.path.join(self.cfg.datasets.configs[0].dataset.path, "calib", "calib.txt"))

        self.train_camera_model = self.train_camera_model.get_scaled_model(self.img_width, self.img_height)

        self.image_to_pointcloud = _ImageToPointcloud(camera_model=self.train_camera_model, device=self.device)

        # -------------------------Source Losses------------------------------------------------------
        try:
            start_decay_epoch = l_n_p[0]['bce']['start_decay_epoch']
            end_decay_epoch = l_n_p[0]['bce']['end_decay_epoch']
            self.print_p_0('Using decaying ratio in BCE Loss')
        except KeyError:
            start_decay_epoch = None
            end_decay_epoch = None
            self.print_p_0('Using constant ratio in BCE Loss')

        self.bce = get_loss('bootstrapped_cross_entropy',
                            img_height=self.img_height,
                            img_width=self.img_width,
                            r=l_n_p[0]['bce']['r'],
                            ignore_index=IGNORE_INDEX_SEMANTIC,
                            start_decay_epoch=start_decay_epoch,
                            end_decay_epoch=end_decay_epoch)

        self.bce_weigth = l_n_w[0]['bce']

        # -------------------------Metrics-for-Validation---------------------------------------------
        assert self.num_classes == 16  # if training on more or less classes please change eval setting for miou

        self.eval_13_classes = [True, True, True, False, False, False, True, True, True, True, True, True, True, True,
                                True, True]

        self.miou_13 = MIoU(num_classes=cfg.datasets.configs[1].dataset.num_classes,
                            ignore_classes=self.eval_13_classes,
                            ignore_index=IGNORE_INDEX_SEMANTIC)
        self.miou_16 = MIoU(num_classes=self.cfg.datasets.configs[1].dataset.num_classes,
                            ignore_index=IGNORE_INDEX_SEMANTIC)
        # -------------------------Initializations----------------------------------------------------

        self.epoch = 0  # current epoch
        self.start_time = None  # time of starting training

    def run(self):
        # if lading from checkpoint set the epoch
        if self.cfg.checkpoint.use_checkpoint:
            self.set_from_checkpoint()
        self.print_p_0(f'Initial epoch {self.epoch}')

        if self.rank == 0:
            for _, net in self.model.get_networks().items():
                wandb.watch(net)

        for self.epoch in range(self.epoch, self.cfg.train.nof_epochs):
            self.print_p_0('Training')
            self.train()
            if self.cfg.val.do_validation:
                self.print_p_0('Validation')
                self.validate()

        self.print_p_0("Training done.")

    def train(self):
        self.set_train()

        # Main loop:
        for batch_idx, data in enumerate(self.train_loader):
            if self.rank == 0:
                self.print_p_0(f"Training epoch {self.epoch} | batch {batch_idx}")
            self.training_step(data, batch_idx)

        # Update the scheduler
        self.scheduler.step()

        # Create checkpoint after each epoch and save it
        if self.rank == 0:
            self.save_checkpoint()

    def validate(self):
        pass  # to be implemented when not using supervised depth for prediction

    def training_step(self, data, batch_idx):
        # Move batch to device
        for key, val in data.items():
            data[key] = val.to(self.device)

        prediction = self.model.forward([self.image_to_pointcloud(data['depth_dense'].double())])[0]

        semantic_pred = prediction['semantic']

        loss, loss_dict = self.compute_losses_source(semantic_pred, data['semantic'])

        self.print_p_0(loss)

        self.optimizer.zero_grad()
        loss.backward()  # mean over individual gpu losses
        self.optimizer.step()

        if self.rank == 0:
            if batch_idx % int(500 / torch.cuda.device_count()) == 0:
                rgb = wandb.Image(data[('rgb', 0)][0].detach().cpu().numpy().transpose(1, 2, 0), caption="Virtual RGB "
                                                                                                         "(Not Used)")
                depth_gt_img = self.get_wandb_depth_image(data['depth_dense'][0], batch_idx)
                sem_img = self.get_wandb_semantic_image(F.softmax(semantic_pred, dim=1)[0], True, 1, f'Semantic Map')
                semantic_gt = self.get_wandb_semantic_image(data['semantic'][0], False, 1,
                                                            f'Semantic GT with id {batch_idx}')
                wandb.log({f'Source images {self.epoch}': [rgb, depth_gt_img, sem_img, semantic_gt]})
            wandb.log({f"total loss epoch {self.epoch}": loss})
            wandb.log(loss_dict)

    def validation_step(self, data, batch_idx):
        pass  # to be implemented when not using supervised depth for prediction

    def compute_losses_source(self, semantic_pred, semantic_gt):
        loss_dict = {}

        soft_semantic_pred = F.softmax(semantic_pred, dim=1)
        bce_loss = self.bce_weigth * self.bce(prediction=soft_semantic_pred, target=semantic_gt, epoch=self.epoch)
        loss_dict['bce'] = bce_loss

        return bce_loss, loss_dict

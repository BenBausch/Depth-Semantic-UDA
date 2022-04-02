# Python modules
import time
import os
import re

import numpy as np
import torch
import matplotlib.pyplot as plt

# Own classes
from utils.plotting_like_cityscapes_utils import visu_depth_prediction as vdp
from losses.metrics import MIoU, DepthEvaluator
from train.base.train_base import TrainSingleDatasetBase
from losses import get_loss
import camera_models
import wandb
import torch.nn.functional as F
from utils.plotting_like_cityscapes_utils import CITYSCAPES_ID_TO_NAME_19, CITYSCAPES_ID_TO_NAME_16
from utils.constans import IGNORE_VALUE_DEPTH


class UnsupervisedDepthTrainer(TrainSingleDatasetBase):
    """
    Used for training on datasets that consist of RGB Images, like Cityscapes Sequence dataset
    """
    def __init__(self, device_id, cfg, world_size=1):
        super(UnsupervisedDepthTrainer, self).__init__(device_id=device_id, cfg=cfg, world_size=world_size)

        # -------------------------Parameters that should be the same for all datasets----------------
        self.num_classes = self.cfg.datasets.configs[0].dataset.num_classes
        for i, dcfg in enumerate(self.cfg.datasets.configs):
            assert self.num_classes == dcfg.dataset.num_classes

        if self.num_classes == 19:
            self.c_id_to_name = CITYSCAPES_ID_TO_NAME_19
        elif self.num_classes == 16:
            self.c_id_to_name = CITYSCAPES_ID_TO_NAME_16
        else:
            raise ValueError("GUDA training not defined for {self.num_classes} classes!")

        # -------------------------Source dataset parameters------------------------------------------
        assert self.cfg.datasets.configs[0].dataset.rgb_frame_offsets[0] == 0, 'RGB offsets must start with 0'

        self.img_width = self.cfg.datasets.configs[0].dataset.feed_img_size[0]
        self.img_height = self.cfg.datasets.configs[0].dataset.feed_img_size[1]
        assert self.img_height % 32 == 0, 'The image height must be a multiple of 32'
        assert self.img_width % 32 == 0, 'The image width must be a multiple of 32'

        self.rgb_frame_offsets = self.cfg.datasets.configs[0].dataset.rgb_frame_offsets
        self.num_input_frames = len(self.cfg.datasets.configs[0].dataset.rgb_frame_offsets)
        self.num_pose_frames = 2 if self.cfg.model.pose_net.input == "pairs" else self.num_input_frames

        l_n_p = self.cfg.datasets.configs[0].losses.loss_names_and_parameters
        l_n_w = self.cfg.datasets.configs[0].losses.loss_names_and_weights

        # Specify whether to train in fully unsupervised manner or not
        if self.cfg.datasets.configs[0].dataset.use_sparse_depth:
            self.print_p_0('Training supervised on source dataset using sparse depth!')
        if self.cfg.datasets.configs[0].dataset.use_dense_depth:
            self.print_p_0('Training supervised on source dataset using dense depth!')
        if self.cfg.datasets.configs[0].dataset.use_self_supervised_depth:
            self.print_p_0('Training unsupervised on source dataset using self supervised depth!')

        self.use_gt_scale_train = self.cfg.datasets.configs[0].eval.train.use_gt_scale and \
                                  self.cfg.datasets.configs[0].eval.train.gt_depth_available
        self.use_gt_scale_val = self.cfg.datasets.configs[0].eval.val.use_gt_scale and \
                                self.cfg.datasets.configs[0].eval.val.gt_depth_available

        if self.use_gt_scale_train:
            self.print_p_0("Source ground truth scale is used for computing depth errors while training.")
        if self.use_gt_scale_val:
            self.print_p_0("Source ground truth scale is used for computing depth errors while validating.")

        self.min_depth = self.cfg.datasets.configs[0].dataset.min_depth
        self.max_depth = self.cfg.datasets.configs[0].dataset.max_depth

        self.use_garg_crop = self.cfg.datasets.configs[0].eval.use_garg_crop

        # Set up normalized camera model for source domain
        try:
            self.train_camera_model = \
                camera_models.get_camera_model_from_file(self.cfg.datasets.configs[0].dataset.camera,
                                                         os.path.join(self.cfg.datasets.configs[0].dataset.path,
                                                                      "calib", "calib.txt"))

        except FileNotFoundError:
            raise Exception("No Camera calibration file found! Create Camera calibration file under: " +
                            os.path.join(self.cfg.datasets.configs[0].dataset.path, "calib", "calib.txt"))

        # -------------------------Source Losses------------------------------------------------------
        self.reconstruction_loss = get_loss('reconstruction',
                                            ref_img_width=self.img_width,
                                            ref_img_height=self.img_height,
                                            normalized_camera_model=self.train_camera_model,
                                            num_scales=self.cfg.model.depth_net.nof_scales,
                                            device=self.device)
        self.reconstruction_use_ssim = l_n_p[0]['reconstruction']['use_ssim']
        self.reconstruction_use_automasking = l_n_p[0]['reconstruction']['use_automasking']
        self.reconstruction_weight = l_n_w[0]['reconstruction']

        self.snr = get_loss('surface_normal_regularization',
                            ref_img_width=self.img_width,
                            ref_img_height=self.img_height,
                            normalized_camera_model=self.train_camera_model,
                            device=self.device)  # used for plotting only

        self.loss_weight = self.cfg.datasets.loss_weights[0]

        # -------------------------Metrics-for-Validation---------------------------------------------
        try:
            self.validation_camera_model = \
                camera_models.get_camera_model_from_file(self.cfg.datasets.configs[1].dataset.camera,
                                                         os.path.join(self.cfg.datasets.configs[1].dataset.path,
                                                                      "calib", "calib.txt"))
        except FileNotFoundError:
            raise Exception("No Camera calibration file found! Create Camera calibration file under: " +
                            os.path.join(self.cfg.datasets.configs[1].dataset.path, "calib", "calib.txt"))

        self.snr_validation = get_loss('surface_normal_regularization',
                                       ref_img_width=self.cfg.datasets.configs[1].dataset.feed_img_size[0],
                                       ref_img_height=self.cfg.datasets.configs[1].dataset.feed_img_size[1],
                                       normalized_camera_model=self.validation_camera_model,
                                       device=self.device)

        self.depth_evaluator = DepthEvaluator(use_garg_crop=self.use_garg_crop)

        # -------------------------Initializations----------------------------------------------------

        self.epoch = 0  # current epoch
        self.start_time = None  # time of starting training
        torch.autograd.set_detect_anomaly(True)

    def run(self):
        # if lading from checkpoint set the epoch
        if self.cfg.checkpoint.use_checkpoint:
            self.set_from_checkpoint()
        self.print_p_0(f'Initial epoch {self.epoch}')

        initial_epoch = self.epoch - 1

        if self.rank == 0:
            for _, net in self.model.get_networks().items():
                wandb.watch(net)

        for self.epoch in range(self.epoch, self.cfg.train.nof_epochs):
            self.print_p_0('Training')
            self.train()
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
        self.set_eval()

        # Main loop:
        for batch_idx, data in enumerate(self.val_loader):
            if self.rank == 0:
                print(f"Evaluation epoch {self.epoch} | batch {batch_idx}")

            self.validation_step(data, batch_idx)

    def training_step(self, data, batch_idx):
        # Move batch to device
        for key, val in data.items():
            data[key] = val.to(self.device)

        prediction = self.model.forward([data])[0]

        pose_pred = prediction['poses']

        depth_pred, raw_sigmoid = prediction['depth'][0], prediction['depth'][1]

        loss, loss_dict = self.compute_losses(data, depth_pred, pose_pred)

        self.print_p_0(loss)

        self.optimizer.zero_grad()
        loss.backward()  # mean over individual gpu losses
        self.optimizer.step()

        # log samples
        if batch_idx % int(5 / torch.cuda.device_count()) == 0 and self.rank == 0:
            rgb = wandb.Image(data[('rgb', 0)][0].detach().cpu().numpy().transpose(1, 2, 0), caption="Virtual RGB")
            depth_img = self.get_wandb_depth_image(depth_pred[('depth', 0)][0].detach(),
                                                   batch_idx)
            normal_img = self.get_wandb_normal_image(depth_pred[('depth', 0)].detach(), self.snr, 'Predicted')
            wandb.log({f'Source images {self.epoch}': [rgb, depth_img, normal_img]})
        if self.rank == 0:
            wandb.log({f"total loss epoch {self.epoch}": loss})
            wandb.log(loss_dict)

    def validation_step(self, data, batch_idx):
        for key, val in data.items():
            data[key] = val.to(self.device)
        prediction = self.model.forward(data, dataset_id=1, predict_depth=True, train=False)[0]
        depth = prediction['depth'][0][('depth', 0)]

        if self.rank == 0 and batch_idx % 15 == 0:
            rgb_img = wandb.Image(data[('rgb', 0)][0].cpu().detach().numpy().transpose(1, 2, 0),
                                  caption=f'Rgb {batch_idx}')
            depth_img = self.get_wandb_depth_image(depth.detach(), batch_idx)
            normal_img = self.get_wandb_normal_image(depth.detach(), self.snr_validation, 'Validation')
            wandb.log(
                {f'images of epoch {self.epoch}': [rgb_img, depth_img, normal_img]})

    def compute_losses(self, data, depth_pred, poses):
        loss_dict = {}
        reconsruction_loss = self.reconstruction_weight * \
                             self.reconstruction_loss(batch_data=data,
                                                      pred_depth=depth_pred,
                                                      poses=poses,
                                                      rgb_frame_offsets=self.cfg.datasets.configs[
                                                          0].dataset.rgb_frame_offsets,
                                                      use_automasking=self.reconstruction_use_automasking,
                                                      use_ssim=self.reconstruction_use_ssim)
        loss_dict[f'reconstruction epoch {self.epoch}'] = reconsruction_loss
        return reconsruction_loss, loss_dict

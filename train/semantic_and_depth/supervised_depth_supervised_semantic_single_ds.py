# Python modules
import time
import os
import re

import numpy as np
import torch
import matplotlib.pyplot as plt

# Own classes
from losses.metrics import MIoU, DepthEvaluator
from train.base.train_base import TrainSingleDatasetBase
from losses import get_loss
import camera_models
import wandb
import torch.nn.functional as F
from utils.plotting_like_cityscapes_utils import CITYSCAPES_ID_TO_NAME_19, CITYSCAPES_ID_TO_NAME_16
from utils.constans import IGNORE_VALUE_DEPTH, IGNORE_INDEX_SEMANTIC


class SemanticDepthTrainer(TrainSingleDatasetBase):
    """
    Trainer to train on datasets with semantic and depth labels, like Synthia
    """
    def __init__(self, device_id, cfg, world_size=1):
        super(SemanticDepthTrainer, self).__init__(device_id=device_id, cfg=cfg, world_size=world_size)

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
        if self.cfg.datasets.configs[0].dataset.use_semantic_gt:
            self.print_p_0('Training supervised on source dataset using semantic annotations!')
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
            self.camera_model = \
                camera_models.get_camera_model_from_file(self.cfg.datasets.configs[0].dataset.camera,
                                                         os.path.join(cfg.datasets.configs[0].dataset.path,
                                                                      "calib", "calib.txt"))

        except FileNotFoundError:
            raise Exception("No Camera calibration file found! Create Camera calibration file under: " +
                            os.path.join(cfg.dataset.path, "calib", "calib.txt"))

        # -------------------------Source Losses------------------------------------------------------
        self.silog_depth = get_loss('silog_depth',
                                    weight=l_n_p[0]['silog_depth']['weight'],
                                    ignore_value=IGNORE_VALUE_DEPTH)

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

        self.snr = get_loss('surface_normal_regularization',
                            ref_img_width=self.img_width,
                            ref_img_height=self.img_height,
                            normalized_camera_model=self.camera_model,
                            device=self.device)

        # get weights for the total loss
        self.silog_depth_weigth = l_n_w[0]['silog_depth']
        self.bce_weigth = l_n_w[0]['bce']
        self.snr_weigth = l_n_w[0]['snr']

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

        mean_iou_13, iou_13 = self.miou_13.get_miou()
        mean_iou_16, iou_16 = self.miou_16.get_miou()

        if self.rank == 0:
            names = [self.c_id_to_name[i] for i in self.c_id_to_name.keys()]
            bar_data = [[label, val] for (label, val) in zip(names, iou_13)]
            table = wandb.Table(data=bar_data, columns=(["Classes", "IOU"]))
            wandb.log({f'IOU per 13 Class epoch {self.epoch}': wandb.plot.bar(table, "Classes", "IOU",
                                                                              title=f"IOU per 13 Class {self.epoch}"),
                       f'Mean IOU per 13 Classes': mean_iou_13})
            names = [self.c_id_to_name[i] for i in self.c_id_to_name.keys()]
            bar_data = [[label, val] for (label, val) in zip(names, iou_16)]
            table = wandb.Table(data=bar_data, columns=(["Classes", "IOU"]))
            wandb.log(
                {f'IOU per 16 Class {self.epoch}': wandb.plot.bar(table, "Classes", "IOU",
                                                                  title=f"IOU per 16 Class {self.epoch}"),
                 f'Mean IOU per 16 Classes': mean_iou_16})

    def training_step(self, data, batch_idx):
        # Move batch to device
        for key, val in data.items():
            data[key] = val.to(self.device)

        prediction = self.model.forward([data])[0]

        depth_pred, raw_sigmoid = prediction['depth'][0][('depth', 0)], prediction['depth'][1][('disp', 0)]

        semantic_pred = prediction['semantic']

        loss, loss_dict = self.compute_losses_source(data['depth_dense'], depth_pred, raw_sigmoid,
                                                     semantic_pred, data['semantic'])

        self.print_p_0(loss)

        self.optimizer.zero_grad()
        loss.backward()  # mean over individual gpu losses
        self.optimizer.step()

        if self.rank == 0:
            if batch_idx % int(500 / torch.cuda.device_count()) == 0:
                depth_errors = '\n Errors avg over this images batch \n' + \
                               self.depth_evaluator.depth_losses_as_string(data['depth_dense'], depth_pred)
                rgb = wandb.Image(data[('rgb', 0)][0].detach().cpu().numpy().transpose(1, 2, 0), caption="Source RGB")
                depth_img = self.get_wandb_depth_image(depth_pred[0].detach(), batch_idx, caption_addon=depth_errors)
                semantic_gt = self.get_wandb_semantic_image(data['semantic'][0], False, 1,
                                                            f'Semantic GT with id {batch_idx}')
                depth_gt_img = self.get_wandb_depth_image(data['depth_dense'][0], batch_idx, caption_addon=depth_errors)
                sem_img = self.get_wandb_semantic_image(F.softmax(semantic_pred, dim=1)[0], True, 1, f'Semantic Map')
                wandb.log({f'Source images {self.epoch}': [rgb, depth_img, depth_gt_img, sem_img, semantic_gt]})
            wandb.log({f"total loss epoch {self.epoch}": loss})
            wandb.log(loss_dict)

    def validation_step(self, data, batch_idx):
        for key, val in data.items():
            data[key] = val.to(self.device)
        prediction = self.model.forward(data, dataset_id=1, predict_depth=True, train=False)[0]

        sem_pred_13 = prediction['semantic']

        soft_pred_13 = F.softmax(sem_pred_13, dim=1)
        self.miou_13.update(mask_pred=soft_pred_13, mask_gt=data['semantic'])

        sem_pred_16 = prediction['semantic']

        soft_pred_16 = F.softmax(sem_pred_16, dim=1)
        self.miou_16.update(mask_pred=soft_pred_16, mask_gt=data['semantic'])

        if self.rank == 0 and batch_idx % 15 == 0:
            rgb_img = wandb.Image(data[('rgb', 0)][0].cpu().detach().numpy().transpose(1, 2, 0),
                                  caption=f'Rgb {batch_idx}')
            semantic_img_13 = self.get_wandb_semantic_image(soft_pred_16[0], True, 1,
                                                            f'Semantic Map image with 16 classes')
            semantic_img_16 = self.get_wandb_semantic_image(soft_pred_16[0], True, 2,
                                                            f'Semantic Map image with 16 classes')
            semantic_gt = self.get_wandb_semantic_image(data['semantic'][0], False, 1,
                                                        f'Semantic GT with id {batch_idx}')
            wandb.log(
                {f'images of epoch {self.epoch}': [rgb_img, semantic_img_13, semantic_img_16, semantic_gt]})

    def compute_losses_source(self, depth_target, depth_pred, raw_sigmoid, semantic_pred, semantic_gt):
        loss_dict = {}
        # inverse depth --> pixels close to the camera have a high value, far away pixels have a small value
        # non-inverse depth map --> pixels far from the camera have a high value, close pixels have a small value

        silog_loss = self.silog_depth_weigth * self.silog_depth(pred=depth_pred,
                                                                target=depth_target)
        loss_dict['silog_loss'] = silog_loss

        soft_semantic_pred = F.softmax(semantic_pred, dim=1)
        bce_loss = self.bce_weigth * self.bce(prediction=soft_semantic_pred, target=semantic_gt, epoch=self.epoch)
        loss_dict['bce'] = bce_loss

        snr_loss = self.snr_weigth * self.snr(depth_prediction=depth_pred, depth_gt=depth_target)
        loss_dict['snr'] = snr_loss

        return silog_loss + bce_loss + snr_loss, loss_dict


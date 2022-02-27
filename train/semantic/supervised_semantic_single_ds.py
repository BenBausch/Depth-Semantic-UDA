# Python modules
import time
import os
import re

import numpy as np
import torch
import matplotlib.pyplot as plt

# Own classes
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


class SupervisedSemanticTrainer(TrainSingleDatasetBase):
    def __init__(self, device_id, cfg, world_size=1):
        super(SupervisedSemanticTrainer, self).__init__(device_id=device_id, cfg=cfg, world_size=world_size)

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
        self.img_width = self.cfg.datasets.configs[0].dataset.feed_img_size[0]
        self.img_height = self.cfg.datasets.configs[0].dataset.feed_img_size[1]
        assert self.img_height % 32 == 0, 'The image height must be a multiple of 32'
        assert self.img_width % 32 == 0, 'The image width must be a multiple of 32'

        l_n_p = self.cfg.datasets.configs[0].losses.loss_names_and_parameters
        l_n_w = self.cfg.datasets.configs[0].losses.loss_names_and_weights

        # Specify whether to train in fully unsupervised manner or not
        if self.cfg.datasets.configs[0].dataset.use_semantic_gt:
            self.print_p_0('Training supervised on source dataset using semantic annotations!')

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

        semantic_pred = prediction['semantic']

        loss, loss_dict = self.compute_losses_source(semantic_pred, data['semantic'])

        self.print_p_0(loss)

        self.optimizer.zero_grad()
        loss.backward()  # mean over individual gpu losses
        self.optimizer.step()

        if self.rank == 0:
            if batch_idx % int(300 / torch.cuda.device_count()) == 0:
                rgb = wandb.Image(data[('rgb', 0)][0].detach().cpu().numpy().transpose(1, 2, 0), caption="Virtual RGB")
                sem_img = self.get_wandb_semantic_image(F.softmax(semantic_pred, dim=1)[0], True, 1, f'Semantic Map')
                semantic_gt = self.get_wandb_semantic_image(data['semantic'][0], False, 1,
                                                            f'Semantic GT with id {batch_idx}')
                wandb.log({f'Virtual images {self.epoch}': [rgb, sem_img, semantic_gt]})
            wandb.log({f'epoch {self.epoch} steps': batch_idx})
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

    def compute_losses_source(self, semantic_pred, semantic_gt):
        loss_dict = {}

        soft_semantic_pred = F.softmax(semantic_pred, dim=1)
        bce_loss = self.bce_weigth * self.bce(prediction=soft_semantic_pred, target=semantic_gt, epoch=self.epoch)
        loss_dict['bce'] = bce_loss

        return bce_loss, loss_dict

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

    def get_wandb_semantic_image(self, semantic, is_prediction=True, k=1, caption=''):
        if is_prediction:
            semantic = torch.topk(semantic, k=k, dim=0, sorted=True).indices[k - 1].unsqueeze(0)
        else:
            semantic = semantic.unsqueeze(0)
        img = wandb.Image(s_to_rgb(semantic.detach().cpu(), num_classes=self.num_classes), caption=caption)
        return img

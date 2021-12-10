# Python modules
# Python modules
import time
import os

import torch

# Own classes
from train.base.train_base import TrainBase
from utils.utils import info_gpu_memory
from utils.losses import get_loss
from io_utils import io_utils
from eval import eval
import torch.nn as nn
import camera_models

#todo: Remove imports below
import matplotlib.pyplot as plt


class SemanticTrainer(TrainBase):
    def __init__(self, cfg):
        super(SemanticTrainer, self).__init__(cfg)

        self.img_width = self.cfg.dataset.feed_img_size[0]
        self.img_height = self.cfg.dataset.feed_img_size[1]
        assert self.img_height % 32 == 0, 'The image height must be a multiple of 32'
        assert self.img_width % 32 == 0, 'The image width must be a multiple of 32'

        assert self.cfg.train.rgb_frame_offsets[0] == 0, 'RGB offsets must start with 0'

        # Initialize some parameters for further usage
        self.num_scales = self.cfg.losses.reconstruction.nof_scales
        self.rgb_frame_offsets = self.cfg.train.rgb_frame_offsets

        self.use_gt_scale_train = self.cfg.eval.train.use_gt_scale and self.cfg.eval.train.gt_depth_available
        self.use_gt_scale_val = self.cfg.eval.val.use_gt_scale and self.cfg.eval.val.gt_depth_available

        if self.use_gt_scale_train:
            print("Ground truth scale is used for computing depth errors while training")
        if self.use_gt_scale_val:
            print("Ground truth scale is used for computing depth errors while validating")

        self.use_garg_crop = self.cfg.eval.use_garg_crop

        # ToDo1: Load model weights if available and wanted...
        # ToDo1: Handle normalized stuff (atm we assume its normalized but don't be sure that it is the case)

        # Get all loss objects for later usage (not all of them may be used...)
        self.crit_ce = get_loss("cross_entropy", ignore_index=250)
        self.soft_max = nn.Softmax()

    def run(self):
        self.epoch = 0
        self.step_train = 0
        self.step_val = 0

        self.start_time = time.time()

        print("Training started...")

        for self.epoch in range(self.cfg.train.nof_epochs):
            self.train()
            self.validate()

        print("Training done.")

    def train(self):

        self.set_train()

        running_loss = 0

        # Main loop:
        for batch_idx, data in enumerate(self.train_loader):
            print("Training epoch {:>3} | batch {:>6}".format(self.epoch, batch_idx))

            # Here also an image dictionary shall be outputted and other stuff to be logged
            loss_value = self.training_step(data, batch_idx)
            print(loss_value)
            running_loss += loss_value

            self.step_train += 1

        print(running_loss / len(self.train_loader))

        # Update the scheduler
        self.scheduler.step()

        # Create checkpoint after each epoch and save it
        self.save_checkpoint()

    def training_step(self, data, batch_idx):

        # Move batch to device
        for key, val in data.items():
            data[key] = val.to(self.device)

        latent = self.model.latent_features(data["rgb", 0])
        prediction = self.model.predict_semantic(latent)

        # Compute the losses
        loss = self.compute_losses(data, prediction)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_losses(self, data, prediction):

        # 1.) Compute the loss of semantic supervision

        loss = self.crit_ce(self.soft_max(prediction), data["gt"].to(dtype=torch.long))

        return loss

    def validate(self):
        self.set_eval()

        # Main loop:
        for batch_idx, data in enumerate(self.val_loader):
            print("Validation epoch {:>3} | batch {:>6}".format(self.epoch, batch_idx))

            self.validation_step(data, batch_idx)

            self.step_val += 1

        self.set_train()

    def validation_step(self, data):

        # Move batch to device
        for key, val in data.items():
            data[key] = val.to(self.device)

        with torch.no_grad():
            latent = self.model.latent_features(data[("rgb", 0)])
            prediction = self.model.predict_semantic(latent)
            loss = self.compute_losses(data, prediction)
            print(loss)

    def set_train(self):
        for m in self.model.networks.values():
            m.train()

    def set_eval(self):
        for m in self.model.networks.values():
            m.eval()

    def set_from_checkpoint(self):
        pass

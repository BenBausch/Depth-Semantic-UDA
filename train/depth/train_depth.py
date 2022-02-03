# Python modules
import time
import os
import re
import torch

# Own classes
from utils.plotting_like_cityscapes_utils import semantic_id_tensor_to_rgb_numpy_array as s_to_rgb
from utils.plotting_like_cityscapes_utils import visu_depth_prediction as vdp
from train.base.train_base import TrainSingleDatasetBase
from losses import get_loss
import camera_models
import wandb
import torch.nn.functional as F


class DepthTrainer(TrainSingleDatasetBase):
    def __init__(self, device_id, cfg, world_size=1):
        super(DepthTrainer, self).__init__(device_id=device_id, cfg=cfg, world_size=world_size)

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
        print(l_n_p)
        l_n_w = self.cfg.datasets.configs[0].losses.loss_names_and_weights
        print(l_n_w)

        # Specify whether to train in fully unsupervised manner or not
        if self.cfg.datasets.configs[0].dataset.use_sparse_depth:
            print('Training supervised on source dataset using sparse depth!')
        if self.cfg.datasets.configs[0].dataset.use_dense_depth:
            print('Training supervised on source dataset using dense depth!')
        if self.cfg.datasets.configs[0].dataset.use_semantic_gt:
            print('Training supervised on source dataset using semantic annotations!')
        if self.cfg.datasets.configs[0].dataset.use_self_supervised_depth:
            print('Training unsupervised on source dataset using self supervised depth!')

        self.use_gt_scale_train = self.cfg.datasets.configs[0].eval.train.use_gt_scale and \
                                         self.cfg.datasets.configs[0].eval.train.gt_depth_available
        self.use_gt_scale_val = self.cfg.datasets.configs[0].eval.val.use_gt_scale and \
                                       self.cfg.datasets.configs[0].eval.val.gt_depth_available

        if self.use_gt_scale_train:
            print("Source ground truth scale is used for computing depth errors while training.")
        if self.use_gt_scale_val:
            print("Source ground truth scale is used for computing depth errors while validating.")

        self.min_depth = self.cfg.datasets.configs[0].dataset.min_depth
        self.max_depth = self.cfg.datasets.configs[0].dataset.max_depth

        self.use_garg_crop = self.cfg.datasets.configs[0].eval.use_garg_crop

        # Set up camera model
        try:
            self.camera_model = \
                camera_models.get_camera_model_from_file(self.cfg.datasets.configs[0].dataset.camera,
                                                         os.path.join(cfg.datasets.configs[0].dataset.path,
                                                                      "calib", "calib.txt"))

        except FileNotFoundError:
            raise Exception("No Camera calibration file found! Create Camera calibration file under: " +
                            os.path.join(cfg.dataset.path, "calib", "calib.txt"))

        # -------------------------Target Losses------------------------------------------------------
        self.reconstruction_loss = get_loss('reconstruction',
                                                   normalized_camera_model=self.camera_model,
                                                   num_scales=self.cfg.model.depth_net.nof_scales,
                                                   device=self.device)
        self.reconstruction_use_ssim = l_n_p[0]['reconstruction']['use_ssim']
        self.reconstruction_use_automasking = l_n_p[0]['reconstruction']['use_automasking']
        self.reconstruction_weight = l_n_w[0]['reconstruction']

        self.epoch = 0  # current epoch
        self.start_time = None  # time of starting training
        self.log_step = 0  # step for each loggin call

    def run(self):
        # if lading from checkpoint set the epoch
        if self.cfg.checkpoint.use_checkpoint:
            self.set_from_checkpoint()
        self.print_p_0(f'Initial epoch {self.epoch}')

        initial_epoch = self.epoch - 1

        if self.rank == 0:
            for _, net in self.model.get_networks().items():
                wandb.watch(net)

        print("Training started...")

        for self.epoch in range(self.epoch, self.cfg.train.nof_epochs):
            self.train()
            print('Validation')
            self.validate()

        print("Training done.")

    def train(self):
        self.set_train()

        # Main loop:
        for batch_idx, data in enumerate(self.train_loader):
            self.log_step += 1
            if self.rank == 0:
                print(f"Training epoch {self.epoch} | batch {batch_idx}")
            self.training_step(data, batch_idx)

        # Update the scheduler
        self.scheduler.step()

        # Create checkpoint after each epoch and save it
        # save only on the first process
        if self.rank == 0:
            self.save_checkpoint()

    def validate(self):
        self.set_eval()

        # Main loop:
        for batch_idx, data in enumerate(self.val_loader):
            self.log_step += 1
            if self.rank == 0:
                print(f"Evaluation epoch {self.epoch} | batch {batch_idx}")

            self.validation_step(data, batch_idx)

    def training_step(self, data, batch_idx):
        # -----------------------------------------------------------------------------------------------
        # ----------------------------------Virtual Sample Processing------------------------------------
        # -----------------------------------------------------------------------------------------------
        # Move batch to device
        for key, val in data.items():
            data[key] = val.to(self.device)
        prediction = self.model.forward([data])[0]  # pass data as list, because of guda model forward implementation for DDP training
        pose_pred = prediction['poses']
        depth_pred, raw_sigmoid = prediction['depth'][0], prediction['depth'][1]
        loss, loss_dict = self.compute_losses(data, depth_pred, pose_pred)

        # log samples
        if batch_idx % 500 == 0 and self.rank == 0:
            rgb = wandb.Image(data[('rgb', 0)][0].detach().cpu().numpy().transpose(1, 2, 0), caption="Real RGB")
            depth_img = self.get_wandb_depth_image(depth_pred[0], batch_idx)
            wandb.log({'Real images': [rgb, depth_img]}, step=self.log_step)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.rank == 0:
            wandb.log({"loss": loss, 'epoch': self.epoch}, step=self.log_step)

    def validation_step(self, data, batch_idx):
        for key, val in data.items():
            data[key] = val.to(self.device)
        prediction = self.model.forward(data, dataset_id=1, predict_depth=True, train=False)[0]
        depth = prediction['depth'][0]

        if self.rank == 0:
            rgb_img = wandb.Image(data[('rgb', 0)][0].cpu().detach().numpy().transpose(1, 2, 0), caption=f'Rgb {batch_idx}')
            depth_img = self.get_wandb_depth_image(depth, batch_idx)
            wandb.log({'Validation Images': [rgb_img, depth_img]}, step=self.log_step)

    def compute_losses(self, data, depth_pred, poses):
        loss_dict = {}

        reconsruction_loss = self.reconstruction_weight * \
                             self.reconstruction_loss(batch_data=data,
                                                             pred_depth=depth_pred,
                                                             poses=poses,
                                                             rgb_frame_offsets=self.rgb_frame_offsets,
                                                             use_automasking=self.reconstruction_use_automasking,
                                                             use_ssim=self.reconstruction_use_ssim)
        loss_dict['reconstruction'] = reconsruction_loss
        return reconsruction_loss, loss_dict

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

    @staticmethod
    def get_wandb_depth_image(depth, batch_idx):
        colormapped_depth = vdp(1 / depth)
        img = wandb.Image(colormapped_depth, caption=f'Depth Map image with id {batch_idx}')
        return img

    @staticmethod
    def get_wandb_semantic_image(semantic, batch_idx):
        semantic = torch.argmax(F.softmax(semantic, dim=0), dim=0).unsqueeze(0)
        img = wandb.Image(s_to_rgb(semantic.detach().cpu()),
                          caption=f'Semantic Map image with id {batch_idx}')
        return img

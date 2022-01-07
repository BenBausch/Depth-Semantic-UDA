# Python modules
import time
import os
import torch

# Own classes
from train.base.train_base import TrainSourceTargetDatasetBase
from losses import get_loss
from io_utils import io_utils
from eval import eval
import camera_models


class GUDATrainer(TrainSourceTargetDatasetBase):
    def __init__(self, cfg):
        super(GUDATrainer, self).__init__(cfg=cfg)

        # -------------------------Source dataset parameters------------------------------------------
        assert self.cfg.datasets.configs[0].dataset.rgb_frame_offsets[0] == 0, 'RGB offsets must start with 0'

        self.source_img_width = self.cfg.datasets.configs[0].dataset.feed_img_size[0]
        self.source_img_height = self.cfg.datasets.configs[0].dataset.feed_img_size[1]
        assert self.source_img_height % 32 == 0, 'The image height must be a multiple of 32'
        assert self.source_img_width % 32 == 0, 'The image width must be a multiple of 32'

        self.source_rgb_frame_offsets = self.cfg.datasets.configs[0].dataset.rgb_frame_offsets
        self.source_num_input_frames = len(self.cfg.datasets.configs[0].dataset.rgb_frame_offsets)
        self.source_num_pose_frames = 2 if self.cfg.model.pose_net.input == "pairs" else self.num_input_frames

        source_l_n_p = self.cfg.datasets.configs[0].losses.loss_names_and_parameters
        print(source_l_n_p)
        source_l_n_w = self.cfg.datasets.configs[0].losses.loss_names_and_weights
        print(source_l_n_w)

        # Specify whether to train in fully unsupervised manner or not
        if self.cfg.datasets.configs[0].dataset.use_sparse_depth:
            print('Training supervised on source dataset using sparse depth!')
        if self.cfg.datasets.configs[0].dataset.use_dense_depth:
            print('Training supervised on source dataset using dense depth!')
        if self.cfg.datasets.configs[0].dataset.use_semantic_gt:
            print('Training supervised on source dataset using semantic annotations!')
        if self.cfg.datasets.configs[0].dataset.use_self_supervised_depth:
            print('Training unsupervised on source dataset using self supervised depth!')

        self.source_use_gt_scale_train = self.cfg.datasets.configs[0].eval.train.use_gt_scale and \
                                         self.cfg.datasets.configs[0].eval.train.gt_depth_available
        self.source_use_gt_scale_val = self.cfg.datasets.configs[0].eval.val.use_gt_scale and \
                                       self.cfg.datasets.configs[0].eval.val.gt_depth_available

        if self.source_use_gt_scale_train:
            print("Source ground truth scale is used for computing depth errors while training.")
        if self.source_use_gt_scale_val:
            print("Source ground truth scale is used for computing depth errors while validating.")

        self.source_min_depth = self.cfg.datasets.configs[0].dataset.min_depth
        self.source_max_depth = self.cfg.datasets.configs[0].dataset.max_depth

        self.source_use_garg_crop = self.cfg.datasets.configs[0].eval.use_garg_crop

        # Set up normalized camera model for source domain
        try:
            self.source_normalized_camera_model = \
                camera_models.get_camera_model_from_file(self.cfg.datasets.configs[0].dataset.camera,
                                                         os.path.join(cfg.datasets.configs[0].dataset.path,
                                                                      "calib", "calib.txt"))

        except FileNotFoundError:
            raise Exception("No Camera calibration file found! Create Camera calibration file under: " +
                            os.path.join(cfg.dataset.path, "calib", "calib.txt"))

        # -------------------------Target dataset parameters------------------------------------------
        assert self.cfg.datasets.configs[1].dataset.rgb_frame_offsets[0] == 0, 'RGB offsets must start with 0'

        self.target_img_width = self.cfg.datasets.configs[1].dataset.feed_img_size[0]
        self.target_img_height = self.cfg.datasets.configs[1].dataset.feed_img_size[1]
        assert self.target_img_height % 32 == 0, 'The image height must be a multiple of 32'
        assert self.target_img_width % 32 == 0, 'The image width must be a multiple of 32'

        target_l_n_p = self.cfg.datasets.configs[1].losses.loss_names_and_parameters
        target_l_n_w = self.cfg.datasets.configs[1].losses.loss_names_and_weights

        # Specify whether to train in fully unsupervised manner or not
        if self.cfg.datasets.configs[1].dataset.use_sparse_depth:
            print('Training supervised on target dataset using sparse depth!')
        if self.cfg.datasets.configs[1].dataset.use_dense_depth:
            print('Training supervised on target dataset using dense depth!')
        if self.cfg.datasets.configs[1].dataset.use_semantic_gt:
            print('Training supervised on target dataset using semantic annotations!')
        if self.cfg.datasets.configs[1].dataset.use_self_supervised_depth:
            print('Training unsupervised on target dataset using self supervised depth!')

        self.target_use_gt_scale_train = self.cfg.datasets.configs[1].eval.train.use_gt_scale and \
                                         self.cfg.datasets.configs[1].eval.train.gt_depth_available
        self.target_use_gt_scale_val = self.cfg.datasets.configs[1].eval.val.use_gt_scale and \
                                       self.cfg.datasets.configs[1].eval.val.gt_depth_available

        if self.target_use_gt_scale_train:
            print("Target ground truth scale is used for computing depth errors while training.")
        if self.target_use_gt_scale_val:
            print("Target ground truth scale is used for computing depth errors while validating.")

        self.target_min_depth = self.cfg.datasets.configs[1].dataset.min_depth
        self.target_max_depth = self.cfg.datasets.configs[1].dataset.max_depth

        self.target_use_garg_crop = self.cfg.datasets.configs[1].eval.use_garg_crop

        # Set up normalized camera model for target domain
        try:
            self.target_normalized_camera_model = \
                camera_models.get_camera_model_from_file(self.cfg.datasets.configs[1].dataset.camera,
                                                         os.path.join(cfg.datasets.configs[1].dataset.path, "calib",
                                                                      "calib.txt"))

        except FileNotFoundError:
            raise Exception("No Camera calibration file found! Create Camera calibration file under: " +
                            os.path.join(cfg.datasets.configs[1].dataset.path, "calib", "calib.txt"))

        # -------------------------Source Losses------------------------------------------------------
        self.source_silog_depth = get_loss('silog_depth',
                                           weight=source_l_n_p[0]['silog_depth']['weight'])

        self.source_bce = get_loss('bootstrapped_cross_entropy',
                                   img_height=self.source_img_height,
                                   img_width=self.source_img_width,
                                   r=source_l_n_p[0]['bce']['r'],
                                   ignore_index=source_l_n_p[0]['bce']['ignore_index'])

        self.source_snr = get_loss('surface_normal_regularization',
                                   normalized_camera_model=self.source_normalized_camera_model,
                                   device=self.device)

        # get weights for the total loss
        self.source_silog_depth_weigth = source_l_n_w[0]['silog_depth']
        self.source_bce_weigth = source_l_n_w[0]['bce']
        self.source_snr_weigth = source_l_n_w[0]['snr']

        self.source_loss_weight = self.cfg.datasets.loss_weights[0]

        # -------------------------Target Losses------------------------------------------------------
        self.target_reconstruction_loss = get_loss('reconstruction',
                                                   normalized_camera_model=self.target_normalized_camera_model,
                                                   num_scales=self.cfg.model.depth_net.nof_scales,
                                                   device=self.device)
        self.target_reconstruction_use_ssim = target_l_n_p[0]['reconstruction']['use_ssim']
        self.target_reconstruction_use_automasking = target_l_n_p[0]['reconstruction']['use_automasking']
        self.target_reconstruction_weight = target_l_n_w[0]['reconstruction']

        self.target_loss_weight = self.cfg.datasets.loss_weights[1]

        # -------------------------Initializations----------------------------------------------------

        self.epoch = None  # current epoch
        self.step_train = None  # current step within training part of epoch
        self.step_val = None  # current step within training part of epoch
        self.start_time = None  # time of starting training

    def run(self):
        self.epoch = 0
        self.step_train = 0
        self.step_val = 0

        self.start_time = time.time()

        print("Training started...")

        for self.epoch in range(self.cfg.train.nof_epochs):
            self.train()

            # self.validate()

        print("Training done.")

    def train(self):

        data = {}

        self.set_train()

        # Main loop:
        for batch_idx, data in enumerate(zip(self.source_train_loader, self.target_train_loader)):
            print(f"Training epoch {self.epoch} | batch {batch_idx}")

            self.training_step(data, batch_idx)


        # Update the scheduler
        self.scheduler.step()

        # Create checkpoint after each epoch and save it
        self.save_checkpoint()

    def training_step(self, data, batch_idx):
        before_op_time = time.time()

        # -----------------------------------------------------------------------------------------------
        # ----------------------------------Virtual Sample Processing------------------------------------
        # -----------------------------------------------------------------------------------------------
        # Move batch to device
        for key, val in data[0].items():
            data[0][key] = val.to(self.device)
        prediction = self.model.forward(data[0], dataset_id=0)

        depth_pred, raw_sigmoid = prediction['depth'][0], prediction['depth'][1]

        semantic_pred = prediction['semantic']

        loss_source = self.compute_losses_source(data[0]['depth_dense'], depth_pred, raw_sigmoid,
                                                 semantic_pred, data[0]['semantic'])

        # -----------------------------------------------------------------------------------------------
        # -----------------------------------Real Sample Processing--------------------------------------
        # -----------------------------------------------------------------------------------------------

        # Move batch to device
        for key, val in data[1].items():
            data[1][key] = val.to(self.device)
        prediction = self.model.forward(data[1], dataset_id=1)

        pose_pred = prediction['poses']

        depth_pred, raw_sigmoid = prediction['depth'][0], prediction['depth'][1]

        loss_target = self.compute_losses_target(data[1], depth_pred, pose_pred)

        # -----------------------------------------------------------------------------------------------
        # ----------------------------------------Optimization-------------------------------------------
        # -----------------------------------------------------------------------------------------------

        loss = self.source_loss_weight * loss_source + self.target_loss_weight * loss_target

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    def compute_losses_source(self, depth_target, depth_pred, raw_sigmoid, semantic_pred, semantic_gt):
        loss_dict = {}
        sillog_loss = self.source_silog_depth_weigth * self.source_silog_depth(pred=depth_pred,
                                                                               target=depth_target)  # todo: get inverse depth
        bce_loss = self.source_bce_weigth * self.source_bce(prediction=semantic_pred, target=semantic_gt)
        snr_loss = self.source_snr_weigth * self.source_snr(depth_prediction=depth_pred, depth_gt=depth_target)

        return sillog_loss + bce_loss + snr_loss

    def compute_losses_target(self, data, depth_pred, poses):
        loss_dict = {}
        reconsruction_loss = \
            self.target_reconstruction_loss(batch_data=data,
                                            pred_depth=depth_pred,
                                            poses=poses,
                                            rgb_frame_offsets=self.cfg.datasets.configs[1].dataset.rgb_frame_offsets,
                                            use_automasking=self.target_reconstruction_use_automasking,
                                            use_ssim=self.target_reconstruction_use_ssim)
        return self.target_reconstruction_weight * reconsruction_loss

    def set_train(self):
        for m in self.model.networks.values():
            m.train()

    def set_eval(self):
        for m in self.model.networks.values():
            m.eval()

    def set_from_checkpoint(self):
        pass

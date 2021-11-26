# Python modules
# Python modules
import time
import os

import torch

# Own classes
from train.train_base import TrainBase
from train.losses import get_loss
from io_utils import io_utils
from eval import eval
import camera_models

class Trainer(TrainBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__(cfg)

        self.img_width = self.cfg.dataset.feed_img_size[0]
        self.img_height = self.cfg.dataset.feed_img_size[1]
        assert self.img_height % 32 == 0, 'The image height must be a multiple of 32'
        assert self.img_width % 32 == 0, 'The image width must be a multiple of 32'

        assert self.cfg.train.rgb_frame_offsets[0] == 0, 'RGB offsets must start with 0'

        # Initialize some parameters for further usage
        self.num_scales = self.cfg.losses.reconstruction.nof_scales
        self.rgb_frame_offsets = self.cfg.train.rgb_frame_offsets
        self.num_input_frames = len(self.cfg.train.rgb_frame_offsets)
        self.num_pose_frames = 2 if self.cfg.model.pose_net.input == "pairs" else self.num_input_frames

        # Specify whether to train in fully unsupervised manner or not
        self.is_unsupervised = not self.cfg.dataset.use_sparse_depth
        if self.is_unsupervised:
            print("Training in fully unsupervised manner. No depth measurements are used!")

        self.use_gt_scale_train = self.cfg.eval.train.use_gt_scale and self.cfg.eval.train.gt_available
        self.use_gt_scale_val = self.cfg.eval.val.use_gt_scale and self.cfg.eval.val.use_gt_scale

        if self.use_gt_scale_train:
            print("Ground truth scale is used for computing depth errors while training")
        if self.use_gt_scale_val:
            print("Ground truth scale is used for computing depth errors while validating")

        self.use_garg_crop = self.cfg.eval.use_garg_crop

        # ToDo: Load model weights if available and wanted...
        # ToDo: Handle normalized stuff (atm we assume its normalized but don't make sure that it is the case)
        # Set up normalized camera model
        self.normalized_camera_model = \
            camera_models.get_camera_model_from_file(self.cfg.dataset.camera, os.path.join(cfg.dataset.path, "calib", "calib.txt"))

        # Get all loss objects for later usage (not all of them may be used...)
        self.crit_l1_depth = get_loss("l1_depth")
        self.crit_depth_reprojection = get_loss("depth_reprojection", self.img_width, self.img_height, self.normalized_camera_model, self.device)
        self.crit_edge_smooth = get_loss("edge_smooth")
        self.crit_reconstruction = get_loss("reconstruction", self.img_width, self.img_height, self.normalized_camera_model, self.num_scales, self.device)

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

        # Main loop:
        for batch_idx, data in enumerate(self.train_loader):
            print("Training epoch {:>3} | batch {:>6}".format(self.epoch, batch_idx))

            # Here also an image dictionary shall be outputted and other stuff to be logged
            self.training_step(data, batch_idx)

            self.step_train += 1

        # Update the scheduler
        self.scheduler.step()

        # Create checkpoint after each epoch and save it
        self.save_checkpoint()

    def training_step(self, data, batch_idx):
        before_op_time = time.time()

        # Move batch to device
        for key, val in data.items():
            data[key] = val.to(self.device)

        # Predict the poses
        poses_pred = self.models.predict_poses(data)

        # ToDo: Use sparse depth data also in the pose prediction maybe? For this we would have to load the data
        # ToDo: Adapt the network for sparse depth inputs
        # Predict the depth map
        input = data["rgb", 0] #if not self.cfg.dataset.use_sparse_depth else torch.cat((data["rgb", 0], data["sparse"]), 1)
        depth_pred, raw_sigmoid = self.models.predict_depth(input)

        # depth_target = data["sparse"] if self.cfg.dataset.use_sparse_depth else data["gt"] # ToDo: Take this in later
        depth_target = data["sparse"] if self.cfg.dataset.use_sparse_depth else None

        # Compute the losses
        losses_dict = self.compute_losses(data, depth_target, depth_pred, raw_sigmoid, poses_pred)

        # Compute the final loss
        loss = losses_dict["photometric_loss"] * self.cfg.losses.weights.reconstruction + losses_dict["smoothness_loss"] * self.cfg.losses.weights.smoothness
        if not self.is_unsupervised:
            loss += losses_dict["depth_loss"] * self.cfg.losses.weights.depth

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log some results from the training
        # The following part is taken from https://github.com/nianticlabs/monodepth2/blob/master/trainer.py
        early_phase = batch_idx % self.cfg.io.log_frequency == 0 and self.step_train < 2000
        late_phase = self.step_train% 2000 == 0

        if early_phase or late_phase:
            # Compute time for processing a batch
            duration_batch = time.time() - before_op_time
            samples_per_sec = self.cfg.train.batch_size / duration_batch
            total_time = time.time() - self.start_time

            # Log and visualize training results for a single batch
            imgs_visu_train = io_utils.IOHandler.generate_imgs_to_visualize(\
                data, depth_pred, poses_pred, self.rgb_frame_offsets, self.crit_reconstruction.image_warpers[0])

            depth_errors = {}
            if "gt" in data:
                _, depth_errors = eval.compute_depth_losses(self.use_gt_scale_train, self.use_garg_crop, data["gt"], depth_pred, gt_size=(self.cfg.dataset.feed_img_size[1], self.cfg.dataset.feed_img_size[0]), depth_ranges=(self.min_depth, self.max_depth))

            # Log relevant data to terminal and tensorboard
            io_utils.IOHandler.log_train(self.writer_train, loss, depth_errors, losses_dict, depth_pred, imgs_visu_train, 4, batch_idx, samples_per_sec, total_time, self.epoch, self.num_total_steps, self.step_train)

    def compute_losses(self, data, depth_target, depth_pred, raw_sigmoid, poses_pred):
        # ToDo: We maybe should compare the inverse depths as their value margin is usually smaller than that of the
        #  depth so, it may be easier to learn it for outdoor/indoor scenarios where the depths ranges are completely
        #  different
        # 1.) Compute the loss by depth supervision
        if depth_target is not None:
            if self.cfg.losses.use_depth_reprojection_loss is False:
                depth_loss = self.crit_l1_depth(depth_pred, depth_target)
            else:
                # Attention: We arbitrarily take the pose prediction from the last frame in the sequence of frames
                # considered (i.e. [0, -1, ->1<-]). You could also use another pose. You could even use an arbitrary
                # pose but then the pose network output does not affect this loss and hence the gradients are not back-
                # propagated to the pose network for this specific loss!
                depth_loss = self.crit_depth_reprojection(depth_pred, depth_target, poses_pred[-1])

        # Attention: We apply the smoothness loss directly on the sigmoid output to avoid that the depth range has an
        # impact on the smoothness loss (e.g. bigger range also scales the derivatives and hence the smoothness loss)
        # 2.) Compute the edge aware smoothness loss
        # Get the normalized inverse depth map
        mean_disp = raw_sigmoid.mean(2, True).mean(3, True)
        norm_disp = raw_sigmoid / (mean_disp + 1e-7)
        smoothness_loss = self.crit_edge_smooth(norm_disp, data["rgb", 0])

        # 3.) Compute the photometric loss
        # ToDo: Nur dort wo keine depth supervision vorliegt... (maybe..) -> Detach the photometric for those pixels with depth supervision??
        photometric_loss = self.crit_reconstruction(data, depth_pred, poses_pred, self.cfg.train.rgb_frame_offsets,
                                                    self.cfg.losses.reconstruction.use_automasking, self.cfg.losses.reconstruction.use_ssim)

        if depth_target is not None:
            loss_dict = {"depth_loss": depth_loss, "photometric_loss": photometric_loss, "smoothness_loss": smoothness_loss}
        else:
            loss_dict = {"photometric_loss": photometric_loss, "smoothness_loss": smoothness_loss}

        print(loss_dict)

        return loss_dict

    def validate(self):
        self.set_eval()

        # Main loop:
        for batch_idx, data in enumerate(self.val_loader):
            print("Validation epoch {:>3} | batch {:>6}".format(self.epoch, batch_idx))

            self.validation_step(data, batch_idx)

            self.step_val += 1

        self.set_train()

    def validation_step(self, data, batch_idx):

        # Move batch to device
        for key, val in data.items():
            data[key] = val.to(self.device)

        with torch.no_grad():
            # Predict the depth map
            imgs = data["rgb", 0]
            depth_pred, _ = self.models.predict_depth(imgs)

            # Compute the depth losses by comparing with gt
            depth_errors = {}
            if "gt" in data:
                _, depth_errors = eval.compute_depth_losses(self.use_gt_scale_val, self.use_garg_crop, data["gt"],
                                                            depth_pred,
                                                            gt_size=(self.cfg.dataset.feed_img_size[1], self.cfg.dataset.feed_img_size[0]),
                                                            depth_ranges=(self.min_depth, self.max_depth))

            # ToDo: You should also consider logging the total loss per epoch as this indicates the actual performance of the network
            #  Based on that the best model so far should be saved as such...
            io_utils.IOHandler.log_val(self.writer_val, depth_errors, data["rgb", 0], depth_pred, 4, self.step_val)


    def set_train(self):
        for m in self.models.networks.values():
            m.train()

    def set_eval(self):
        for m in self.models.networks.values():
            m.eval()

    def set_from_checkpoint(self):
        pass


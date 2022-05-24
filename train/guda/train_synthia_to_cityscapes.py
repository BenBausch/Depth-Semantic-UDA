# Python modules
import time
import os
import re
import torch

# Own classes
from utils.plotting_like_cityscapes_utils import semantic_id_tensor_to_rgb_numpy_array as s_to_rgb
from utils.plotting_like_cityscapes_utils import CITYSCAPES_ID_TO_NAME_19, CITYSCAPES_ID_TO_NAME_16
from utils.plotting_like_cityscapes_utils import visu_depth_prediction as vdp
from losses.metrics import MIoU, DepthEvaluator
from train.base.train_base import TrainSourceTargetDatasetBase
from losses import get_loss
import camera_models
import wandb
import torch.nn.functional as F
from utils.constans import IGNORE_VALUE_DEPTH, IGNORE_INDEX_SEMANTIC
from helper_modules.pseudo_label_fusion import fuse_pseudo_labels_with_gorund_truth


class GUDATrainer(TrainSourceTargetDatasetBase):
    """
    Trainer to run the guda algorithm for synthia to cityscapes.
    """

    def __init__(self, device_id, cfg, world_size=1):
        super(GUDATrainer, self).__init__(device_id=device_id, cfg=cfg, world_size=world_size)

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
        assert self.cfg.datasets.configs[0].dataset.rgb_frame_offsets[0] == 0, 'RGB offsets must start with 0'

        self.source_img_width = self.cfg.datasets.configs[0].dataset.feed_img_size[0]
        self.source_img_height = self.cfg.datasets.configs[0].dataset.feed_img_size[1]
        assert self.source_img_height % 32 == 0, 'The image height must be a multiple of 32'
        assert self.source_img_width % 32 == 0, 'The image width must be a multiple of 32'

        self.source_rgb_frame_offsets = self.cfg.datasets.configs[0].dataset.rgb_frame_offsets
        self.source_num_input_frames = len(self.cfg.datasets.configs[0].dataset.rgb_frame_offsets)
        self.source_num_pose_frames = 2 if self.cfg.model.pose_net.input == "pairs" else self.num_input_frames

        source_l_n_p = self.cfg.datasets.configs[0].losses.loss_names_and_parameters
        self.print_p_0(source_l_n_p)
        source_l_n_w = self.cfg.datasets.configs[0].losses.loss_names_and_weights
        self.print_p_0(source_l_n_w)

        # Specify whether to train in fully unsupervised manner or not
        if self.cfg.datasets.configs[0].dataset.use_sparse_depth:
            self.print_p_0('Training supervised on source dataset using sparse depth!')
        if self.cfg.datasets.configs[0].dataset.use_dense_depth:
            self.print_p_0('Training supervised on source dataset using dense depth!')
        if self.cfg.datasets.configs[0].dataset.use_semantic_gt:
            self.print_p_0('Training supervised on source dataset using semantic annotations!')
        if self.cfg.datasets.configs[0].dataset.use_self_supervised_depth:
            self.print_p_0('Training unsupervised on source dataset using self supervised depth!')

        self.source_use_gt_scale_train = self.cfg.datasets.configs[0].eval.train.use_gt_scale and \
                                         self.cfg.datasets.configs[0].eval.train.gt_depth_available
        self.source_use_gt_scale_val = self.cfg.datasets.configs[0].eval.val.use_gt_scale and \
                                       self.cfg.datasets.configs[0].eval.val.gt_depth_available

        if self.source_use_gt_scale_train:
            self.print_p_0("Source ground truth scale is used for computing depth errors while training.")
        if self.source_use_gt_scale_val:
            self.print_p_0("Source ground truth scale is used for computing depth errors while validating.")

        self.source_min_depth = self.cfg.datasets.configs[0].dataset.min_depth
        self.source_max_depth = self.cfg.datasets.configs[0].dataset.max_depth

        self.source_use_garg_crop = self.cfg.datasets.configs[0].eval.use_garg_crop
        if self.source_use_garg_crop:
            self.print_p_0(f'Use Garg Crop for depth evaluation on source dataset!')

        # Set up normalized camera model for source domain
        try:
            self.source_normalized_camera_model = \
                camera_models.get_camera_model_from_file(self.cfg.datasets.configs[0].dataset.camera,
                                                         os.path.join(cfg.datasets.configs[0].dataset.path,
                                                                      "calib", "calib.txt"))

        except FileNotFoundError:
            raise Exception("No Camera calibration file found! Create Camera calibration file under: " +
                            os.path.join(cfg.datasets.configs[0].dataset.path, "calib", "calib.txt"))

        # -------------------------Target dataset parameters------------------------------------------
        assert self.cfg.datasets.configs[1].dataset.rgb_frame_offsets[0] == 0, 'RGB offsets must start with 0'

        self.target_img_width = self.cfg.datasets.configs[1].dataset.feed_img_size[0]
        self.target_img_height = self.cfg.datasets.configs[1].dataset.feed_img_size[1]
        assert self.target_img_height % 32 == 0, 'The image height must be a multiple of 32'
        assert self.target_img_width % 32 == 0, 'The image width must be a multiple of 32'

        self.target_rgb_frame_offsets = self.cfg.datasets.configs[1].dataset.rgb_frame_offsets
        target_l_n_p = self.cfg.datasets.configs[1].losses.loss_names_and_parameters
        target_l_n_w = self.cfg.datasets.configs[1].losses.loss_names_and_weights

        # Specify whether to train in fully unsupervised manner or not
        if self.cfg.datasets.configs[1].dataset.use_sparse_depth:
            self.print_p_0('Training supervised on target dataset using sparse depth!')
        if self.cfg.datasets.configs[1].dataset.use_dense_depth:
            self.print_p_0('Training supervised on target dataset using dense depth!')
        if self.cfg.datasets.configs[1].dataset.use_semantic_gt:
            self.print_p_0('Training supervised on target dataset using semantic annotations!')
        if self.cfg.datasets.configs[1].dataset.use_self_supervised_depth:
            self.print_p_0('Training unsupervised on target dataset using self supervised depth!')

        self.target_use_gt_scale_train = self.cfg.datasets.configs[1].eval.train.use_gt_scale and \
                                         self.cfg.datasets.configs[1].eval.train.gt_depth_available
        self.target_use_gt_scale_val = self.cfg.datasets.configs[1].eval.val.use_gt_scale and \
                                       self.cfg.datasets.configs[1].eval.val.gt_depth_available

        if self.target_use_gt_scale_train:
            self.print_p_0("Target ground truth scale is used for computing depth errors while training.")
        if self.target_use_gt_scale_val:
            self.print_p_0("Target ground truth scale is used for computing depth errors while validating.")

        self.target_min_depth = self.cfg.datasets.configs[1].dataset.min_depth
        self.target_max_depth = self.cfg.datasets.configs[1].dataset.max_depth

        self.target_use_garg_crop = self.cfg.datasets.configs[1].eval.use_garg_crop

        self.target_predict_semantic_for_whole_sequence = \
            self.cfg.datasets.configs[1].dataset.predict_semantic_for_each_img_in_sequence

        # Set up normalized camera model for target domain
        try:
            self.target_normalized_camera_model = \
                camera_models.get_camera_model_from_file(self.cfg.datasets.configs[1].dataset.camera,
                                                         os.path.join(cfg.datasets.configs[1].dataset.path, "calib",
                                                                      "calib.txt"))

        except FileNotFoundError:
            raise Exception("No Camera calibration file found! Create Camera calibration file under: " +
                            os.path.join(cfg.datasets.configs[1].dataset.path, "calib", "calib.txt"))

        # -------------------------Mixed Dataset------------------------------------------------------
        if self.use_mixed_dataset:
            assert self.cfg.datasets.configs[3].dataset.rgb_frame_offsets[0] == 0, 'RGB offsets must start with 0'

            self.mixed_img_width = self.cfg.datasets.configs[3].dataset.feed_img_size[0]
            self.mixed_img_height = self.cfg.datasets.configs[3].dataset.feed_img_size[1]
            assert self.mixed_img_height % 32 == 0, 'The image height must be a multiple of 32'
            assert self.mixed_img_width % 32 == 0, 'The image width must be a multiple of 32'

            self.mixed_rgb_frame_offsets = self.cfg.datasets.configs[3].dataset.rgb_frame_offsets
            mixed_l_n_p = self.cfg.datasets.configs[3].losses.loss_names_and_parameters
            mixed_l_n_w = self.cfg.datasets.configs[3].losses.loss_names_and_weights

            self.correct_dataset_id_mapping = {0: 0, 1: 1,
                                               2: 3}  # used to get the correct configuration when predicting
            # in guda, the validation configuration should be skipped during training
        else:
            self.correct_dataset_id_mapping = None

        # -------------------------Source Losses------------------------------------------------------
        self.source_silog_depth = get_loss('silog_depth',
                                           weight=source_l_n_p[0]['silog_depth']['weight'],
                                           ignore_value=IGNORE_VALUE_DEPTH)

        try:
            start_decay_epoch = source_l_n_p[0]['bce']['start_decay_epoch']
            end_decay_epoch = source_l_n_p[0]['bce']['end_decay_epoch']
            self.print_p_0('Using decaying ratio in BCE Loss')
        except KeyError:
            start_decay_epoch = None
            end_decay_epoch = None
            self.print_p_0('Using constant ratio in BCE Loss')

        self.source_bce = get_loss('bootstrapped_cross_entropy',
                                   img_height=self.source_img_height,
                                   img_width=self.source_img_width,
                                   r=source_l_n_p[0]['bce']['r'],
                                   ignore_index=IGNORE_INDEX_SEMANTIC,
                                   start_decay_epoch=start_decay_epoch,
                                   end_decay_epoch=end_decay_epoch)

        self.source_snr = get_loss('surface_normal_regularization',
                                   ref_img_width=self.source_img_width,
                                   ref_img_height=self.source_img_height,
                                   normalized_camera_model=self.source_normalized_camera_model,
                                   device=self.device)

        # get weights for the total loss
        self.source_silog_depth_weigth = source_l_n_w[0]['silog_depth']
        self.source_bce_weigth = source_l_n_w[0]['bce']
        self.source_snr_weigth = source_l_n_w[0]['snr']

        self.source_loss_weight = self.cfg.datasets.loss_weights[0]

        # -------------------------Target Losses------------------------------------------------------
        self.target_edge_smoothness_loss = get_loss("edge_smooth")
        self.target_reconstruction_loss = get_loss('reconstruction',
                                                   ref_img_width=self.target_img_width,
                                                   ref_img_height=self.target_img_height,
                                                   normalized_camera_model=self.target_normalized_camera_model,
                                                   num_scales=self.cfg.model.depth_net.nof_scales,
                                                   device=self.device)
        self.target_reconstruction_use_ssim = target_l_n_p[0]['reconstruction']['use_ssim']
        self.target_reconstruction_use_automasking = target_l_n_p[0]['reconstruction']['use_automasking']

        # if using tscl then also need to predict semantics for all frames.
        if 'temporal_semantic_consistency' in target_l_n_w[0].keys():
            self.target_temporal_semantic_consistency_weigth = target_l_n_w[0]['temporal_semantic_consistency']
            assert target_l_n_p[0]['reconstruction']['use_temporal_semantic_consistency'] == \
                   self.target_predict_semantic_for_whole_sequence, \
                'If the model should predict semantics the whole target sequence, ' \
                'if should also be used in the target loss! Otherwise unused predictions'
            self.target_use_tsc = True
        else:
            self.target_use_tsc = False

        # either use both motion regularization losses or none
        assert ('motion_group_smoothness' in target_l_n_w[0].keys()) == ('motion_sparsity' in target_l_n_w[0].keys())
        if 'motion_group_smoothness' in target_l_n_w[0].keys():
            self.target_motion_group_smoothness_loss = get_loss("motion_group_smoothness")
            self.target_motion_sparsity_loss = get_loss("motion_sparsity")
            self.target_motion_group_smoothness_weight = target_l_n_w[0]['motion_group_smoothness']
            self.target_motion_sparsity_weight = target_l_n_w[0]['motion_sparsity']

        self.target_reconstruction_weight = target_l_n_w[0]['reconstruction']
        self.target_edge_smoothness_weight = target_l_n_w[0]['edge_smooth']

        self.target_loss_weight = self.cfg.datasets.loss_weights[1]

        # -------------------------Mixed Dataset Losses-----------------------------------------------
        if self.use_mixed_dataset:
            self.mixed_cross_entopy_loss = get_loss("cross_entropy",
                                                    ignore_index=IGNORE_INDEX_SEMANTIC,
                                                    reduction='none')
            self.mixed_cross_entopy_weight = mixed_l_n_w[0]['cross_entropy']
            self.mixed_pseudo_label_threshold = mixed_l_n_w[0]['pseudo_label_threshold']

        # -------------------------Metrics-for-Validation---------------------------------------------
        assert self.num_classes == 16  # if training on more or less classes please change eval setting for miou

        self.eval_13_classes = [True, True, True, False, False, False, True, True, True, True, True, True, True, True,
                                True, True]

        self.miou_13 = MIoU(num_classes=cfg.datasets.configs[2].dataset.num_classes,
                            ignore_classes=self.eval_13_classes,
                            ignore_index=IGNORE_INDEX_SEMANTIC)
        self.miou_16 = MIoU(num_classes=self.cfg.datasets.configs[2].dataset.num_classes,
                            ignore_index=IGNORE_INDEX_SEMANTIC)

        # the target domain should have same resolution as the validation dataset
        # both should also use the same camera intrinsics, since the same snr_validation module will be used for
        # plotting surface normals of target and validation dataset (This is not used during training!)
        assert self.cfg.datasets.configs[2].dataset.feed_img_size[0] == self.target_img_width
        assert self.cfg.datasets.configs[2].dataset.feed_img_size[1] == self.target_img_height

        try:
            self.validation_normalized_camera_model = \
                camera_models.get_camera_model_from_file(self.cfg.datasets.configs[2].dataset.camera,
                                                         os.path.join(cfg.datasets.configs[2].dataset.path, "calib",
                                                                      "calib.txt"))

        except FileNotFoundError:
            raise Exception("No Camera calibration file found! Create Camera calibration file under: " +
                            os.path.join(cfg.datasets.configs[2].dataset.path, "calib", "calib.txt"))

        self.snr_validation = get_loss('surface_normal_regularization',
                                       ref_img_width=self.cfg.datasets.configs[2].dataset.feed_img_size[0],
                                       ref_img_height=self.cfg.datasets.configs[2].dataset.feed_img_size[1],
                                       normalized_camera_model=self.validation_normalized_camera_model,
                                       device=self.device)

        self.depth_evaluator = DepthEvaluator(use_garg_crop=self.source_use_garg_crop)

        # -------------------------Initializations----------------------------------------------------

        self.epoch = 0  # current epoch
        self.start_time = None  # time of starting training
        self.iteration_step = 0

    def run(self):
        # if lading from checkpoint set the epoch
        if self.cfg.checkpoint.use_checkpoint:
            self.set_from_checkpoint()
        self.print_p_0(f'Initial epoch {self.epoch}')

        if self.rank == 0:
            for _, net in self.model.get_networks().items():
                wandb.watch(net)

        self.print_p_0("Training started...")

        for self.epoch in range(self.epoch, self.cfg.train.nof_epochs):
            self.print_p_0('Training')
            self.train()
            if self.cfg.val.do_validation:
                self.print_p_0('Validation')
                self.validate()

        self.print_p_0("Training done.")

    def train(self):
        self.set_train()

        if self.use_mixed_dataset:
            loaders_zip = zip(self.source_train_loader, self.target_train_loader, self.mixed_train_loader)
        else:
            loaders_zip = zip(self.source_train_loader, self.target_train_loader)
        # Main loop:
        for batch_idx, data in enumerate(loaders_zip):
            if self.rank == 0:
                self.print_p_0(f"Training epoch {self.epoch} | batch {batch_idx}")
            self.training_step(data, batch_idx)
            self.iteration_step += 1

        # Update the scheduler
        self.scheduler.step()

        # Create checkpoint after each epoch and save it
        if self.rank == 0:
            self.save_checkpoint()

    def validate(self):
        self.set_eval()

        # Main loop:
        for batch_idx, data in enumerate(self.target_val_loader):
            if self.rank == 0:
                self.print_p_0(f"Evaluation epoch {self.epoch} | batch {batch_idx}")

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
        for key, val in data[0].items():
            data[0][key] = val.to(self.device)
        for key, val in data[1].items():
            data[1][key] = val.to(self.device)
        if self.use_mixed_dataset:
            for key, val in data[2].items():
                data[2][key] = val.to(self.device)
        # -----------------------------------------------------------------------------------------------
        # ----------------------------------Virtual Sample Processing------------------------------------
        # -----------------------------------------------------------------------------------------------
        prediction = self.model.forward(data, correct_dataset_id_mapping=self.correct_dataset_id_mapping)

        prediction_s = prediction[0]

        depth_pred_s, raw_sigmoid_s = prediction_s['depth'][0][('depth', 0)], prediction_s['depth'][1][('disp', 0)]

        semantic_pred_s = prediction_s['semantic']

        loss_source, loss_source_dict = self.compute_losses_source(data[0]['depth_dense'], depth_pred_s, raw_sigmoid_s,
                                                                   semantic_pred_s, data[0]['semantic'])

        # -----------------------------------------------------------------------------------------------
        # -----------------------------------Real Sample Processing--------------------------------------
        # -----------------------------------------------------------------------------------------------

        # Move batch to device
        prediction_t = prediction[1]

        pose_pred_t = prediction_t['poses']

        depth_pred_t, raw_sigmoid_t = prediction_t['depth'][0], prediction_t['depth'][1][('disp', 0)]

        object_motion_map = prediction_t['motion']
        if object_motion_map is not None:
            motion_map = {}
            if object_motion_map is not None:
                for offset in self.target_rgb_frame_offsets[1:]:
                    if self.iteration_step < 0:
                        object_motion_map[offset] = torch.zeros_like(object_motion_map[offset])
                    motion_map[offset] = torch.clone(object_motion_map[offset])
                    motion_map[offset][:, 0, :, :] += pose_pred_t[offset][0, 0, 3]
                    motion_map[offset][:, 1, :, :] += pose_pred_t[offset][0, 1, 3]
                    motion_map[offset][:, 2, :, :] += pose_pred_t[offset][0, 2, 3]
        else:
            motion_map = None

        if self.target_predict_semantic_for_whole_sequence:
            semantic_sequence = {}
            for offset in prediction_t['semantic_sequence'].keys():
                semantic_sequence[offset] = F.softmax(prediction_t['semantic_sequence'][offset], dim=1)
        else:
            semantic_sequence = None

        loss_target, loss_target_dict, warped_imgs = self.compute_losses_target(data[1],
                                                                                depth_pred_t,
                                                                                pose_pred_t,
                                                                                raw_sigmoid_t,
                                                                                semantic_sequence,
                                                                                object_motion_map,
                                                                                motion_map)
        warped_imgs = warped_imgs["rgb"]

        loss = self.source_loss_weight * loss_source + self.target_loss_weight * loss_target

        # -----------------------------------------------------------------------------------------------
        # --------------------------------Augmented Sample Processing------------------------------------
        # -----------------------------------------------------------------------------------------------
        if self.use_mixed_dataset:
            pseudo_labels_m = self.ema_model.forward(data[2], dataset_id=3,
                                                 predict_pseudo_labels_only=True)[0]['pseudo_labels']
            pseudo_labels_m = F.softmax(pseudo_labels_m, dim=1)

            semantic_pred_m = F.softmax(prediction[2]['semantic'], dim=1)

            labels_m, pixel_wise_weight = fuse_pseudo_labels_with_gorund_truth(
                pseudo_label_prediction=pseudo_labels_m,
                ground_turth_labels=data[2]["semantic"])

            mixed_loss = self.mixed_cross_entopy_weight * \
                         torch.mean(pixel_wise_weight * self.mixed_cross_entopy_loss(input=semantic_pred_m,
                                                                                     target=labels_m))
            loss_target_dict[
                "mixed_cross_entropy_loss"] = mixed_loss  # hacky to include in the target_loss_dict, but no
            # problem since just for plotting
            loss += mixed_loss
        # -----------------------------------------------------------------------------------------------
        # ----------------------------------------Optimization-------------------------------------------
        # -----------------------------------------------------------------------------------------------

        self.print_p_0(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.use_mixed_dataset:
            self.optimizer.zero_grad()
            self.update_ema_model_copy(self.iteration_step, 0.99)

        # log samples
        if batch_idx % int(500 / torch.cuda.device_count()) == 0 and self.rank == 0:
            depth_errors = '\n Errors avg over this images batch \n' + \
                           self.depth_evaluator.depth_losses_as_string(data[0]['depth_dense'], depth_pred_s)
            rgb_0 = wandb.Image(data[0][('rgb', 0)][0].detach().cpu().numpy().transpose(1, 2, 0), caption="Source RGB")
            depth_img_0 = self.get_wandb_depth_image(depth_pred_s[0].detach(), batch_idx, caption_addon=depth_errors)
            depth_gt_img_0 = self.get_wandb_depth_image(data[0]['depth_dense'][0], batch_idx,
                                                        caption_addon=depth_errors)
            sem_img_0 = self.get_wandb_semantic_image(F.softmax(semantic_pred_s, dim=1)[0], True, 1, f'Semantic Map')
            semantic_gt_0 = self.get_wandb_semantic_image(data[0]['semantic'][0], False, 1, f'Semantic GT {batch_idx}')
            normal_img_0 = self.get_wandb_normal_image(depth_pred_s.detach(), self.source_snr, 'Predicted')
            normal_gt_img_0 = self.get_wandb_normal_image(data[0]['depth_dense'], self.source_snr, 'GT')
            wandb.log({f'Source Images {self.epoch}': [rgb_0,
                                                       depth_img_0, depth_gt_img_0,
                                                       sem_img_0, semantic_gt_0,
                                                       normal_img_0, normal_gt_img_0]})

            # target plotting
            rgb_1 = wandb.Image(data[1][('rgb', 0)][0].detach().cpu().numpy().transpose(1, 2, 0), caption="Target RGB")
            depth_img_1 = self.get_wandb_depth_image(depth_pred_t[('depth', 0)][0], batch_idx)
            normal_img_1 = self.get_wandb_normal_image(depth_pred_t[('depth', 0)].detach(), self.snr_validation,
                                                       'Predicted')

            prev_warped_img = wandb.Image(warped_imgs[-1].cpu().numpy().transpose(1, 2, 0), caption="Warped RGB -1")
            succ_warped_img = wandb.Image(warped_imgs[+1].cpu().numpy().transpose(1, 2, 0), caption="Warped RGB +1")
            if object_motion_map is not None:
                prev_obj_motion = wandb.Image(object_motion_map[-1][0], caption='frame -1 to frame 0 obj motion')
                succ_obj_motion = wandb.Image(object_motion_map[1][0], caption='frame 0 to frame 1 obj motion')
                prev_motion = wandb.Image(motion_map[-1][0], caption='frame -1 to frame 0 motion')
                succ_motion = wandb.Image(motion_map[1][0], caption='frame 0 to frame 1 motion')
                wandb.log({f'Target images {self.epoch}': [rgb_1, depth_img_1,
                                                           normal_img_1,
                                                           prev_warped_img,
                                                           prev_motion,
                                                           prev_obj_motion,
                                                           succ_warped_img,
                                                           succ_motion,
                                                           succ_obj_motion]})
            else:
                wandb.log({f'Target images {self.epoch}': [rgb_1,
                                                           depth_img_1,
                                                           normal_img_1,
                                                           prev_warped_img,
                                                           succ_warped_img]})

            # mixed dataset plotting
            if self.use_mixed_dataset:
                rgb_2 = wandb.Image(data[2][('rgb', 0)][0].detach().cpu().numpy().transpose(1, 2, 0),
                                    caption="Source RGB")
                sem_img_2 = self.get_wandb_semantic_image(semantic_pred_m[0], True, 1, f'Semantic Map')
                semantic_gt_2 = self.get_wandb_semantic_image(labels_m[0], False, 1, f'Semantic GT {batch_idx}')
                wandb.log({f'Mixed Dataset Images {self.epoch}': [rgb_2, sem_img_2, semantic_gt_2]})

        if self.rank == 0:
            wandb.log({f"total loss epoch {self.epoch}": loss})
            wandb.log(loss_source_dict)
            wandb.log(loss_target_dict)

    def validation_step(self, data, batch_idx):
        for key, val in data.items():
            data[key] = val.to(self.device)
        prediction = self.model.forward(data, dataset_id=2, predict_depth=True, train=False)[0]
        depth = prediction['depth'][0][('depth', 0)].cpu().detach()

        sem_pred_13 = prediction['semantic']

        soft_pred_13 = F.softmax(sem_pred_13, dim=1)
        self.miou_13.update(mask_pred=soft_pred_13, mask_gt=data['semantic'])

        sem_pred_16 = prediction['semantic']

        soft_pred_16 = F.softmax(sem_pred_16, dim=1)
        self.miou_16.update(mask_pred=soft_pred_16, mask_gt=data['semantic'])

        if self.rank == 0 and batch_idx % 15 == 0:
            rgb_img = wandb.Image(data[('rgb', 0)][0].cpu().detach().numpy().transpose(1, 2, 0),
                                  caption=f'Rgb {batch_idx}')
            depth_img = self.get_wandb_depth_image(depth, batch_idx)
            semantic_img_13 = self.get_wandb_semantic_image(soft_pred_16[0], True, 1,
                                                            f'Semantic Map image with 16 classes')
            semantic_img_2nd = self.get_wandb_semantic_image(soft_pred_16[0], True, 2,
                                                             f'Second Highest Predictions')
            semantic_gt = self.get_wandb_semantic_image(data['semantic'][0], False, 1,
                                                        f'Semantic GT with id {batch_idx}')
            wandb.log(
                {f'images of epoch {self.epoch}': [rgb_img, depth_img, semantic_img_13, semantic_img_2nd, semantic_gt]})

    def compute_losses_source(self, depth_target, depth_pred, raw_sigmoid, semantic_pred, semantic_gt):
        loss_dict = {}
        # inverse depth --> pixels close to the camera have a high value, far away pixels have a small value
        # non-inverse depth map --> pixels far from the camera have a high value, close pixels have a small value

        silog_loss = self.source_silog_depth_weigth * self.source_silog_depth(pred=depth_pred,
                                                                              target=depth_target)
        loss_dict[f'silog_loss epoch {self.epoch}'] = silog_loss

        soft_semantic_pred = F.softmax(semantic_pred, dim=1)
        bce_loss = self.source_bce_weigth * self.source_bce(prediction=soft_semantic_pred, target=semantic_gt,
                                                            epoch=self.epoch)
        loss_dict[f'bce epoch {self.epoch}'] = bce_loss

        snr_loss = self.source_snr_weigth * self.source_snr(depth_prediction=depth_pred, depth_gt=depth_target)
        loss_dict[f'snr epoch {self.epoch}'] = snr_loss

        return silog_loss + bce_loss + snr_loss, loss_dict

    def compute_losses_target(self, data, depth_pred, poses, raw_sigmoid, semantic_sequence, object_motion_map, motion_map):
        loss_dict = {}
        reconsruction_loss, semantic_consistency_loss, warped_imgs = self.target_reconstruction_loss(
            batch_data=data,
            pred_depth=depth_pred,
            poses=poses,
            rgb_frame_offsets=
            self.cfg.datasets.configs[
                1].dataset.rgb_frame_offsets,
            use_automasking=self.target_reconstruction_use_automasking,
            use_ssim=self.target_reconstruction_use_ssim,
            semantic_logits=semantic_sequence,
            motion_map=motion_map,
            return_warped_images=True)
        reconsruction_loss = reconsruction_loss * self.target_reconstruction_weight
        loss_dict[f'Reconstruction epoch {self.epoch}'] = reconsruction_loss

        sum = reconsruction_loss

        if self.target_use_tsc:
            semantic_consistency_loss = semantic_consistency_loss * \
                                        self.target_temporal_semantic_consistency_weigth
            sum += semantic_consistency_loss
            loss_dict[f'Temporal Semantic Consistency epoch {self.epoch}'] = semantic_consistency_loss

        if object_motion_map is not None:
            motion_sum = []
            print('Using Motion Regularization!')
            for offset in self.target_rgb_frame_offsets[1:]:
                normalized_obj_motion = object_motion_map[
                    offset]  # normalize_motion_map(object_motion_map[offset], motion_map[offset])
                motion_regularization = self.target_motion_group_smoothness_loss(normalized_obj_motion) * \
                                        self.target_motion_group_smoothness_weight
                motion_regularization += self.target_motion_sparsity_loss(normalized_obj_motion) * \
                                         self.target_motion_sparsity_weight
                motion_sum.append(motion_regularization)
            motion_sum = torch.sum(torch.stack(motion_sum))
            sum += motion_sum
            loss_dict[f'Motion Regularization epoch {self.epoch}'] = motion_sum

        mean_disp = raw_sigmoid.mean(2, True).mean(3, True)
        norm_disp = raw_sigmoid / (mean_disp + 1e-7)
        smoothness_loss = self.target_edge_smoothness_weight * \
                          self.target_edge_smoothness_loss(norm_disp, data["rgb", 0])
        loss_dict[f'Edge Smoothess epoch {self.epoch}'] = smoothness_loss
        sum += smoothness_loss

        return sum, loss_dict, warped_imgs


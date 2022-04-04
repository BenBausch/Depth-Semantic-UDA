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


class SupervisedSemanticUnsupervisedDepthTrainer(TrainSourceTargetDatasetBase):
    """
    Trainer for training supervised on semantic dataset and unsupervised on sequence dataset. For example:
    cityscapes_semantic and cityscapes_sequence
    """

    def __init__(self, device_id, cfg, world_size=1):
        super(SupervisedSemanticUnsupervisedDepthTrainer, self).__init__(device_id=device_id, cfg=cfg,
                                                                         world_size=world_size)
        # renaming of dataloaders to make it less confusing
        self.semantic_train_loader, self.semantic_num_train_files = \
            self.source_train_loader, self.source_num_train_files
        self.sequence_train_loader, self.sequence_num_train_files = \
            self.target_train_loader, self.target_num_train_files
        self.semantic_validation_loader, self.semantic_num_validation_files = \
            self.target_val_loader, self.target_num_val_files

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

        # -------------------------semantic dataset parameters------------------------------------------
        assert self.cfg.datasets.configs[0].dataset.rgb_frame_offsets[0] == 0, \
            'RGB offsets must start with 0'
        assert len(self.cfg.datasets.configs[0].dataset.rgb_frame_offsets) == 1, \
            'Semantic dataset should not be sequence dataset'

        self.semantic_img_width = self.cfg.datasets.configs[0].dataset.feed_img_size[0]
        self.semantic_img_height = self.cfg.datasets.configs[0].dataset.feed_img_size[1]
        assert self.semantic_img_height % 32 == 0, 'The image height must be a multiple of 32'
        assert self.semantic_img_width % 32 == 0, 'The image width must be a multiple of 32'

        semantic_l_n_p = self.cfg.datasets.configs[0].losses.loss_names_and_parameters
        print(semantic_l_n_p)
        semantic_l_n_w = self.cfg.datasets.configs[0].losses.loss_names_and_weights
        print(semantic_l_n_w)

        # Specify whether to train in fully unsupervised manner or not
        if self.cfg.datasets.configs[0].dataset.use_semantic_gt:
            print('Training supervised on semantic dataset using semantic annotations!')

        # -------------------------sequence dataset parameters------------------------------------------

        assert self.cfg.datasets.configs[1].dataset.rgb_frame_offsets[0] == 0, 'RGB offsets must start with 0'

        self.sequence_img_width = self.cfg.datasets.configs[1].dataset.feed_img_size[0]
        self.sequence_img_height = self.cfg.datasets.configs[1].dataset.feed_img_size[1]
        assert self.sequence_img_height % 32 == 0, 'The image height must be a multiple of 32'
        assert self.sequence_img_width % 32 == 0, 'The image width must be a multiple of 32'

        sequence_l_n_p = self.cfg.datasets.configs[1].losses.loss_names_and_parameters
        print(sequence_l_n_p)
        sequence_l_n_w = self.cfg.datasets.configs[1].losses.loss_names_and_weights
        print(sequence_l_n_w)

        # Specify whether to train in fully unsupervised manner or not
        if self.cfg.datasets.configs[1].dataset.use_sparse_depth:
            print('Training supervised on sequence dataset using sparse depth!')
        if self.cfg.datasets.configs[1].dataset.use_dense_depth:
            print('Training supervised on sequence dataset using dense depth!')
        if self.cfg.datasets.configs[1].dataset.use_self_supervised_depth:
            print('Training unsupervised on sequence dataset using self supervised depth!')

        self.sequence_use_gt_scale_train = self.cfg.datasets.configs[1].eval.train.use_gt_scale and \
                                           self.cfg.datasets.configs[1].eval.train.gt_depth_available
        self.sequence_use_gt_scale_val = self.cfg.datasets.configs[1].eval.val.use_gt_scale and \
                                         self.cfg.datasets.configs[1].eval.val.gt_depth_available

        if self.sequence_use_gt_scale_train:
            print("sequence ground truth scale is used for computing depth errors while training.")
        if self.sequence_use_gt_scale_val:
            print("sequence ground truth scale is used for computing depth errors while validating.")

        self.sequence_min_depth = self.cfg.datasets.configs[1].dataset.min_depth
        self.sequence_max_depth = self.cfg.datasets.configs[1].dataset.max_depth

        self.sequence_use_garg_crop = self.cfg.datasets.configs[1].eval.use_garg_crop

        # Set up normalized camera model for sequence domain
        try:
            self.sequence_normalized_camera_model = \
                camera_models.get_camera_model_from_file(self.cfg.datasets.configs[1].dataset.camera,
                                                         os.path.join(cfg.datasets.configs[1].dataset.path, "calib",
                                                                      "calib.txt"))

        except FileNotFoundError:
            raise Exception("No Camera calibration file found! Create Camera calibration file under: " +
                            os.path.join(cfg.datasets.configs[1].dataset.path, "calib", "calib.txt"))

        # -------------------------semantic Losses------------------------------------------------------
        try:
            start_decay_epoch = semantic_l_n_p[0]['bce']['start_decay_epoch']
            end_decay_epoch = semantic_l_n_p[0]['bce']['end_decay_epoch']
            self.print_p_0('Using decaying ratio in BCE Loss')
        except KeyError:
            start_decay_epoch = None
            end_decay_epoch = None
            self.print_p_0('Using constant ratio in BCE Loss')

        self.semantic_bce = get_loss('bootstrapped_cross_entropy',
                                     img_height=self.semantic_img_height,
                                     img_width=self.semantic_img_width,
                                     r=semantic_l_n_p[0]['bce']['r'],
                                     ignore_index=IGNORE_INDEX_SEMANTIC,
                                     start_decay_epoch=start_decay_epoch,
                                     end_decay_epoch=end_decay_epoch)

        # get weights for the total loss
        self.semantic_bce_weigth = semantic_l_n_w[0]['bce']

        self.semantic_loss_weight = self.cfg.datasets.loss_weights[0]

        # -------------------------sequence Losses------------------------------------------------------
        self.sequence_edge_smoothness_loss = get_loss("edge_smooth")
        self.sequence_reconstruction_loss = get_loss('reconstruction',
                                                     ref_img_width=self.sequence_img_width,
                                                     ref_img_height=self.sequence_img_height,
                                                     normalized_camera_model=self.sequence_normalized_camera_model,
                                                     num_scales=self.cfg.model.depth_net.nof_scales,
                                                     device=self.device)
        self.sequence_reconstruction_use_ssim = sequence_l_n_p[0]['reconstruction']['use_ssim']
        self.sequence_reconstruction_use_automasking = sequence_l_n_p[0]['reconstruction']['use_automasking']
        self.sequence_reconstruction_weight = sequence_l_n_w[0]['reconstruction']
        self.sequence_edge_smoothness_weight = sequence_l_n_w[0]['edge_smooth']

        self.sequence_loss_weight = self.cfg.datasets.loss_weights[1]

        # snr is only used to plot the surface normals (make the quality of the depth maps more visible)
        self.snr = get_loss('surface_normal_regularization',
                            ref_img_width=self.sequence_img_width,
                            ref_img_height=self.sequence_img_height,
                            normalized_camera_model=self.sequence_normalized_camera_model,
                            device=self.device)

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

        print("Training started...")

        for self.epoch in range(self.epoch, self.cfg.train.nof_epochs):
            self.train()
            print('Validation')
            self.validate()

        print("Training done.")

    def train(self):
        self.set_train()

        # Main loop:
        for batch_idx, data in enumerate(zip(self.semantic_train_loader, self.sequence_train_loader)):
            if self.rank == 0:
                print(f"Training epoch {self.epoch} | batch {batch_idx}")
            self.training_step(data, batch_idx)

        # Update the scheduler
        self.scheduler.step()

        # Create checkpoint after each epoch and save it
        if self.rank == 0:
            self.save_checkpoint()

    def validate(self):
        self.set_eval()

        # Main loop:
        for batch_idx, data in enumerate(self.semantic_validation_loader):
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
        # -----------------------------------------------------------------------------------------------
        # ----------------------------------Virtual Sample Processing------------------------------------
        # -----------------------------------------------------------------------------------------------
        # Move batch to device
        for key, val in data[0].items():
            data[0][key] = val.to(self.device)
        for key, val in data[1].items():
            data[1][key] = val.to(self.device)
        prediction = self.model.forward(data)

        semantic_pred = prediction[0]['semantic']

        loss_semantic, loss_semantic_dict = self.compute_losses_semantic(semantic_pred, data[0]['semantic'])

        # -----------------------------------------------------------------------------------------------
        # -----------------------------------Real Sample Processing--------------------------------------
        # -----------------------------------------------------------------------------------------------

        # Move batch to device
        prediction_t = prediction[1]

        pose_pred = prediction_t['poses']

        depth_pred, raw_sigmoid = prediction_t['depth'][0], prediction_t['depth'][1][('disp', 0)]

        loss_sequence, loss_sequence_dict = self.compute_losses_sequence(data[1], depth_pred, pose_pred, raw_sigmoid)

        # -----------------------------------------------------------------------------------------------
        # ----------------------------------------Optimization-------------------------------------------
        # -----------------------------------------------------------------------------------------------

        loss = self.semantic_loss_weight * loss_semantic + self.sequence_loss_weight * loss_sequence

        print(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # -----------------------------------------------------------------------------------------------
        # ------------------------------------Plotting and Logging---------------------------------------
        # -----------------------------------------------------------------------------------------------
        if batch_idx % int(500 / torch.cuda.device_count()) == 0 and self.rank == 0:
            # semantic
            rgb_0 = wandb.Image(data[0][('rgb', 0)][0].detach().cpu().numpy().transpose(1, 2, 0), caption="Train RGB")
            sem_img_0 = self.get_wandb_semantic_image(F.softmax(semantic_pred, dim=1)[0], True, 1, f'Semantic Map')
            semantic_gt_0 = self.get_wandb_semantic_image(data[0]['semantic'][0], False, 1,
                                                          f'Semantic GT with id {batch_idx}')
            wandb.log({f'Epoch {self.epoch} Semantic Train Dataset Images': [rgb_0, sem_img_0, semantic_gt_0]})

            # sequence
            rgb_1 = wandb.Image(data[1][('rgb', 0)][0].detach().cpu().numpy().transpose(1, 2, 0), caption="Source RGB")
            depth_img_1 = self.get_wandb_depth_image(depth_pred[('depth', 0)].detach(), batch_idx)
            normal_img_1 = self.get_wandb_normal_image(depth_pred[('depth', 0)].detach(), self.snr, 'Predicted')
            wandb.log({f'Epoch {self.epoch} Sequence Train Dataset Images': [rgb_1, depth_img_1, normal_img_1]})

        if self.rank == 0:
            wandb.log({f"total loss epoch {self.epoch}": loss})
            wandb.log(loss_semantic_dict)
            wandb.log(loss_sequence_dict)

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

    def compute_losses_semantic(self, semantic_pred, semantic_gt):
        loss_dict = {}
        # inverse depth --> pixels close to the camera have a high value, far away pixels have a small value
        # non-inverse depth map --> pixels far from the camera have a high value, close pixels have a small value
        soft_semantic_pred = F.softmax(semantic_pred, dim=1)
        bce_loss = self.semantic_bce_weigth * self.semantic_bce(prediction=soft_semantic_pred, target=semantic_gt,
                                                                epoch=self.epoch)
        loss_dict[f'bce epoch {self.epoch}'] = bce_loss

        return bce_loss, loss_dict

    def compute_losses_sequence(self, data, depth_pred, poses, raw_sigmoid):
        loss_dict = {}
        reconsruction_loss, _ = self.sequence_reconstruction_weight * \
                                self.sequence_reconstruction_loss(
                                    batch_data=data,
                                    pred_depth=depth_pred,
                                    poses=poses,
                                    rgb_frame_offsets=self.cfg.datasets.configs[
                                        1].dataset.rgb_frame_offsets,
                                    use_automasking=self.sequence_reconstruction_use_automasking,
                                    use_ssim=self.sequence_reconstruction_use_ssim)[0]
        loss_dict[f'reconstruction epoch {self.epoch}'] = reconsruction_loss

        mean_disp = raw_sigmoid.mean(2, True).mean(3, True)
        norm_disp = raw_sigmoid / (mean_disp + 1e-7)
        smoothness_loss = self.sequence_edge_smoothness_weight * \
                          self.sequence_edge_smoothness_loss(norm_disp, data["rgb", 0])
        loss_dict[f'Edge smoothess epoch {self.epoch}'] = smoothness_loss

        return reconsruction_loss + smoothness_loss, loss_dict

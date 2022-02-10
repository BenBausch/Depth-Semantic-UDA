# Python modules
import time
import os
import re
import torch

# Own classes
from utils.plotting_like_cityscapes_utils import semantic_id_tensor_to_rgb_numpy_array as s_to_rgb
from utils.plotting_like_cityscapes_utils import CITYSCAPES_ID_TO_NAME
from utils.plotting_like_cityscapes_utils import visu_depth_prediction as vdp
from losses.metrics import MIoU, DepthEvaluator
from train.base.train_base import TrainSourceTargetDatasetBase
from losses import get_loss
import camera_models
import wandb
import torch.nn.functional as F
from utils.constans import IGNORE_VALUE_DEPTH, IGNORE_INDEX_SEMANTIC


class GUDATrainer(TrainSourceTargetDatasetBase):
    def __init__(self, device_id, cfg, world_size=1):
        super(GUDATrainer, self).__init__(device_id=device_id, cfg=cfg, world_size=world_size)

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
        if self.source_use_garg_crop:
            print(f'Use Garg Crop for depth evaluation on source dataset!')

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
                                           weight=source_l_n_p[0]['silog_depth']['weight'],
                                           ignore_value=IGNORE_VALUE_DEPTH)

        self.source_bce = get_loss('bootstrapped_cross_entropy',
                                   img_height=self.source_img_height,
                                   img_width=self.source_img_width,
                                   r=source_l_n_p[0]['bce']['r'],
                                   ignore_index=IGNORE_INDEX_SEMANTIC)

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

        # -------------------------Metrics-for-Validation---------------------------------------------
        self.eval_13_classes = [True, True, True, False, False, False, True, True, True, False, True, True, True, True,
                                False, True, False, True, True]
        self.not_eval_13_classes = [not cl for cl in self.eval_13_classes]
        self.eval_16_classes = [True, True, True, True, True, True, True, True, True, False, True, True, True, True,
                                False, True, False, True, True]
        self.not_eval_16_classes = [not cl for cl in self.eval_16_classes]
        self.miou_13 = MIoU(num_classes=cfg.datasets.configs[2].dataset.num_classes,
                            ignore_classes=self.eval_13_classes,
                            ignore_index=IGNORE_INDEX_SEMANTIC)
        self.miou_16 = MIoU(num_classes=self.cfg.datasets.configs[2].dataset.num_classes,
                            ignore_classes=self.eval_16_classes,
                            ignore_index=IGNORE_INDEX_SEMANTIC)

        self.depth_evaluator = DepthEvaluator(use_garg_crop=self.source_use_garg_crop)

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

        print("Training started...")

        for self.epoch in range(self.epoch, self.cfg.train.nof_epochs):
            self.train()
            print('Validation')
            self.validate()

        print("Training done.")

    def train(self):
        self.set_train()

        # Main loop:
        for batch_idx, data in enumerate(zip(self.source_train_loader, self.target_train_loader)):
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
        for batch_idx, data in enumerate(self.target_val_loader):
            if self.rank == 0:
                print(f"Evaluation epoch {self.epoch} | batch {batch_idx}")

            self.validation_step(data, batch_idx)

        mean_iou_13, iou_13 = self.miou_13.get_miou()
        mean_iou_16, iou_16 = self.miou_16.get_miou()

        if self.rank == 0:
            names = [CITYSCAPES_ID_TO_NAME[i] for i in CITYSCAPES_ID_TO_NAME.keys()]
            bar_data = [[label, val] for (label, val) in zip(names, iou_13)]
            table = wandb.Table(data=bar_data, columns=(["Classes", "IOU"]))
            wandb.log({f'IOU per 13 Class epoch {self.epoch}': wandb.plot.bar(table, "Classes", "IOU",
                                                                              title="IOU per 13 Class"),
                       f'Mean IOU per 13 Classes {self.epoch}': mean_iou_13})
            names = [CITYSCAPES_ID_TO_NAME[i] for i in CITYSCAPES_ID_TO_NAME.keys()]
            bar_data = [[label, val] for (label, val) in zip(names, iou_16)]
            table = wandb.Table(data=bar_data, columns=(["Classes", "IOU"]))
            wandb.log(
                {f'IOU per 16 Class {self.epoch}': wandb.plot.bar(table, "Classes", "IOU", title="IOU per 16 Class"),
                 f'Mean IOU per 16 Classes {self.epoch}': mean_iou_16})

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

        prediction_s = prediction[0]

        depth_pred, raw_sigmoid = prediction_s['depth'][0], prediction_s['depth'][1]

        semantic_pred = prediction_s['semantic']

        loss_source, loss_source_dict = self.compute_losses_source(data[0]['depth_dense'], depth_pred, raw_sigmoid,
                                                                   semantic_pred, data[0]['semantic'])

        # log samples
        if batch_idx % int(50 / torch.cuda.device_count()) == 0 and self.rank == 0:
            depth_errors = '\n Errors avg over this images batch \n' + \
                           self.depth_evaluator.depth_losses_as_string(data[0]['depth_dense'], depth_pred)
            rgb = wandb.Image(data[0][('rgb', 0)][0].detach().cpu().numpy().transpose(1, 2, 0), caption="Virtual RGB")
            depth_img = self.get_wandb_depth_image(depth_pred[0], batch_idx, caption_addon=depth_errors)
            depth_gt_img = self.get_wandb_depth_image(data[0]['depth_dense'][0], batch_idx, caption_addon=depth_errors)
            sem_img = self.get_wandb_semantic_image(F.softmax(semantic_pred, dim=1)[0], True, 1, f'Semantic Map')
            wandb.log({f'Virtual images {self.epoch}': [rgb, depth_img, sem_img]})

        # -----------------------------------------------------------------------------------------------
        # -----------------------------------Real Sample Processing--------------------------------------
        # -----------------------------------------------------------------------------------------------

        # Move batch to device
        prediction_t = prediction[1]

        pose_pred = prediction_t['poses']

        depth_pred, raw_sigmoid = prediction_t['depth'][0], prediction_t['depth'][1]

        loss_target, loss_target_dict = self.compute_losses_target(data[1], depth_pred, pose_pred)

        # log samples
        if batch_idx % int(50 / torch.cuda.device_count()) == 0 and self.rank == 0:
            rgb = wandb.Image(data[1][('rgb', 0)][0].detach().cpu().numpy().transpose(1, 2, 0), caption="Real RGB")
            depth_img = self.get_wandb_depth_image(depth_pred[0], batch_idx)
            wandb.log({f'Real images {self.epoch}': [rgb, depth_img]})

        # -----------------------------------------------------------------------------------------------
        # ----------------------------------------Optimization-------------------------------------------
        # -----------------------------------------------------------------------------------------------

        loss = self.source_loss_weight * loss_source + self.target_loss_weight * loss_target

        print(loss)

        self.optimizer.zero_grad()
        loss.backward()  # mean over individual gpu losses
        self.optimizer.step()

        if self.rank == 0:
            wandb.log({f'epoch {self.epoch} steps': batch_idx})
            wandb.log({f"total loss epoch {self.epoch}": loss})
            wandb.log(loss_source_dict)
            wandb.log(loss_target_dict)

    def validation_step(self, data, batch_idx):
        for key, val in data.items():
            data[key] = val.to(self.device)
        prediction = self.model.forward(data, dataset_id=2, predict_depth=True, train=False)[0]
        depth = prediction['depth'][0]

        # zero out the probabilities of the classes that are not considered
        # zeroing out instead of training on subset of classes permits to train single model for the 13 and 16 class
        # evaluation instead of two models
        sem_pred_13 = prediction['semantic']
        sem_pred_13[:, self.not_eval_13_classes, :, :] = 0.0

        soft_pred_13 = F.softmax(sem_pred_13, dim=1)
        self.miou_13.update(mask_pred=soft_pred_13, mask_gt=data['semantic'])

        sem_pred_16 = prediction['semantic']
        sem_pred_16[:, self.not_eval_16_classes, :, :] = 0.0

        soft_pred_16 = F.softmax(sem_pred_16, dim=1)
        self.miou_16.update(mask_pred=soft_pred_16, mask_gt=data['semantic'])

        if self.rank == 0:
            rgb_img = wandb.Image(data[('rgb', 0)][0].cpu().detach().numpy().transpose(1, 2, 0),
                                  caption=f'Rgb {batch_idx}')
            depth_img = self.get_wandb_depth_image(depth, batch_idx)

            semantic_img_13 = self.get_wandb_semantic_image(soft_pred_13[0], True, 1,
                                                            f'Semantic Map image with 13 classes')

            semantic_img_16 = self.get_wandb_semantic_image(soft_pred_16[0], True, 1,
                                                            f'Semantic Map image with 16 classes')
            semantic_gt = self.get_wandb_semantic_image(data['semantic'][0], False, 1,
                                                        f'Semantic GT with id {batch_idx}')

            wandb.log(
                {f'images of epoch {self.epoch}': [rgb_img, depth_img, semantic_img_13, semantic_img_16, semantic_gt]})

    def compute_losses_source(self, depth_target, depth_pred, raw_sigmoid, semantic_pred, semantic_gt):
        loss_dict = {}
        # inverse depth --> pixels close to the camera have a high value, far away pixels have a small value
        # non-inverse depth map --> pixels far from the camera have a high value, close pixels have a small value

        silog_loss = self.source_silog_depth_weigth * self.source_silog_depth(pred=depth_pred,
                                                                              target=depth_target)
        loss_dict[f'silog_loss epoch {self.epoch}'] = silog_loss

        soft_semantic_pred = F.softmax(semantic_pred, dim=1)
        bce_loss = self.source_bce_weigth * self.source_bce(prediction=soft_semantic_pred, target=semantic_gt)
        loss_dict[f'bce epoch {self.epoch}'] = bce_loss

        snr_loss = self.source_snr_weigth * self.source_snr(depth_prediction=depth_pred, depth_gt=depth_target)
        loss_dict[f'snr epoch {self.epoch}'] = snr_loss

        return silog_loss + bce_loss + snr_loss, loss_dict

    def compute_losses_target(self, data, depth_pred, poses):
        loss_dict = {}
        reconsruction_loss = self.target_reconstruction_weight * \
                             self.target_reconstruction_loss(batch_data=data,
                                                             pred_depth=depth_pred,
                                                             poses=poses,
                                                             rgb_frame_offsets=self.cfg.datasets.configs[
                                                                 1].dataset.rgb_frame_offsets,
                                                             use_automasking=self.target_reconstruction_use_automasking,
                                                             use_ssim=self.target_reconstruction_use_ssim)
        loss_dict[f'reconstruction epoch {self.epoch}'] = reconsruction_loss
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
    def get_wandb_depth_image(depth, batch_idx, caption_addon=''):
        colormapped_depth = vdp(1 / depth)
        img = wandb.Image(colormapped_depth, caption=f'Depth Map of {batch_idx}'+caption_addon)
        return img

    @staticmethod
    def get_wandb_semantic_image(semantic, is_prediction=True, k=1, caption=''):
        if is_prediction:
            semantic = torch.topk(semantic, k=k, dim=0, sorted=True).indices[k - 1].unsqueeze(0)
        else:
            semantic = semantic.unsqueeze(0)
        img = wandb.Image(s_to_rgb(semantic.detach().cpu()),
                          caption=caption)
        return img

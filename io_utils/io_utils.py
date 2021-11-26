import os
import json

import time

import torch
from tensorboardX import SummaryWriter

import matplotlib as mpl
from matplotlib import pyplot as plt

from yacs.config import CfgNode as CN
from cfg.config import get_cfg_defaults

# For visualization
import numpy as np
import PIL.Image as pil

class IOHandler:
    def __init__(self, path_base):
        self.path_base = path_base

    # ToDo: Specify the best model and save it as such
    def save_checkpoint(self, info, checkpoint):
        # Create folder
        path_save_folder = os.path.join(self.path_base, info["time"] + "_" + info["dataset"],  "checkpoints")

        if not os.path.exists(path_save_folder):
            os.makedirs(path_save_folder)

        # Create filename
        filename_save_file = "checkpoint_epoch_{}.pth".format(info["epoch"])

        # Save checkpoint
        path_save = os.path.join(path_save_folder, filename_save_file)
        print("Saving checkpoint for epoch {}".format(info["epoch"]))
        torch.save(checkpoint, path_save)

        # Give only read/execute permissions to everyone for the files created
        os.chmod(path_save, 0o555)

    def save_cfg(self, info, cfg):
        """
        Source: https://github.com/nianticlabs/monodepth2/blob/master/trainer.py
        Save options to disk so we know what we ran this experiment with
        """
        # Create folder
        path_save_folder = os.path.join(self.path_base, info["time"] + "_" + info["dataset"])

        # to_save = cfg.dump().__dict__.copy()

        if not os.path.exists(path_save_folder):
            os.makedirs(path_save_folder)

        to_save = dict(cfg)

        with open(os.path.join(path_save_folder, 'cfg.yaml'), 'w') as f:
            json.dump(to_save, f, indent=2)

    @staticmethod
    def load_cfg(path_cfg):
        cfg = get_cfg_defaults()
        cfg.merge_from_file(path_cfg)
        cfg.freeze()
        return cfg

    @staticmethod
    def load_weights(checkpoint, models, optimizer=None):
        # Go through each model and update weights with those stored in the checkpoint
        for model_name, model in models.items():
            print("Loading weights of {}".format(model_name))
            model.load_state_dict(checkpoint[model_name])

            for key_item_1, key_item_2 in zip(model.state_dict().items(), checkpoint[model_name].items()):
                assert torch.equal(key_item_1[1], key_item_2[1]), \
                    "Mismatch found concerning model weights after loading!"
                assert key_item_1[0] == key_item_2[0], \
                    "Mismatch found concerning model keys after loading. Check whether you load the correct model."

        if optimizer is not None:
            # Load also the corresponding optimizer
            optimizer.load_state_dict(checkpoint["optimizer"])

        return checkpoint

    @staticmethod
    def gen_checkpoint(models, **kwargs):
        checkpoint = {k: val for k, val in kwargs.items()}
        models_dict = {k: val.state_dict() for k, val in models.items()}
        checkpoint.update(models_dict)
        return checkpoint

    @staticmethod
    def load_checkpoint(path_base_checkpoint, filename_checkpoint):
        path_checkpoint = os.path.join(path_base_checkpoint, 'checkpoints', filename_checkpoint)
        assert os.path.exists(path_checkpoint), "The path {} does not exist!".format(path_checkpoint)
        checkpoint = torch.load(path_checkpoint)
        return checkpoint

    @staticmethod
    def log_train(writer, total_loss, depth_errors, losses, depth_pred, images_dict, max_imgs, batch_idx, samples_per_sec, total_time, epoch, num_total_steps, step):
        IOHandler._log_images(writer, images_dict, max_imgs, step)
        IOHandler._log_losses(writer, depth_errors, step)
        IOHandler._log_losses(writer, losses, step)
        IOHandler._log_depth_prediction(writer, depth_pred, max_imgs, step)
        IOHandler._log_visu_depth_prediction(writer, 1.0/depth_pred, max_imgs, step)
        IOHandler._log_cmd(batch_idx, samples_per_sec, total_time, total_loss, losses, epoch, num_total_steps, step)

    @staticmethod
    def log_val(writer, depth_errors, rgb, depth_pred, max_imgs, step):
        # Log imgs
        writer.add_images("rgb", rgb.data, step)
        # Log losses and the depth prediction
        IOHandler._log_losses(writer, depth_errors, step)
        IOHandler._log_depth_prediction(writer, depth_pred, max_imgs, step)
        IOHandler._log_visu_depth_prediction(writer, 1.0/depth_pred, max_imgs, step)

    @staticmethod
    def _log_losses(writer, losses, step):
        for loss_type, loss in losses.items():
            writer.add_scalar("{}".format(loss_type), loss, step)

    @staticmethod
    def _log_images(writer, images_dict, max_imgs, step):
        for identifier, batch_imgs in images_dict.items():
            writer.add_images(identifier, batch_imgs[:max_imgs].data, step)

    @staticmethod
    def _log_depth_prediction(writer, depth_map, max_imgs, step):
        max = float(depth_map.max().cpu().data)
        min = float(depth_map.min().cpu().data)
        diff = max - min if max != min else 1e5
        norm_depth_map = (depth_map - min) / diff
        writer.add_images("depth_pred", norm_depth_map[:max_imgs], step)

    @staticmethod
    def _log_visu_depth_prediction(writer, inv_depth_pred, max_imgs, step):
        batch = []
        for i in range(min(len(inv_depth_pred), max_imgs)):
            disp_np = inv_depth_pred[i, :, :, :].squeeze().cpu().detach().numpy()
            vmax = np.percentile(disp_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
            mapper = plt.cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_img = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
            batch.append(colormapped_img)

        imgs = np.stack(batch, axis=0)
        imgs_permuted = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        writer.add_images("depth_pred_visu", imgs_permuted, step)

    """Adapted from: https://github.com/nianticlabs/monodepth2/blob/master/trainer.py"""
    @staticmethod
    def _log_cmd(batch_idx, samples_per_sec, total_time, total_loss, losses, epoch, num_total_steps, step):
        time_left = (num_total_steps / step - 1.0) * total_time if step > 0 else 0

        print_string =  "Training epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | total loss (weighted) {:.5f} | photom loss {:.5f}" + " | depth loss {}" \
            " | smooth loss {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(epoch, batch_idx, samples_per_sec, total_loss, losses["photometric_loss"], losses["depth_loss"] if "depth_loss" in losses else "N/A",
                                  losses["smoothness_loss"], IOHandler.sec_to_hm_str(total_time), IOHandler.sec_to_hm_str(time_left)))

    @staticmethod
    def sec_to_hm_str(t):
        """
        Source: https://github.com/nianticlabs/monodepth2/blob/master/utils.py
        Convert time in seconds to a nice string
        e.g. 10239 -> '02h50m39s'
        """
        h, m, s = IOHandler.sec_to_hm(t)
        return "{:02d}h{:02d}m{:02d}s".format(h, m, s)

    @staticmethod
    def sec_to_hm(t):
        """
        Source: https://github.com/nianticlabs/monodepth2/blob/master/utils.py
        Convert time in seconds to time in hours, minutes and seconds
        e.g. 10239 -> (2, 50, 39)
        """
        t = int(t)
        s = t % 60
        t //= 60
        m = t % 60
        t //= 60
        return t, m, s

    @staticmethod
    def generate_imgs_to_visualize(batch_data, pred_depth, poses, rgb_frame_offsets, image_warper):
        imgs_to_visualize = {}
        imgs_to_visualize["orig_id_0"] = batch_data[("rgb", 0)]

        for frame_id in rgb_frame_offsets[1:]:
            adjacent_img = batch_data[("rgb", frame_id)]
            imgs_to_visualize["orig_id_{}".format(frame_id)] = adjacent_img

            warped_adjacent_img = image_warper(adjacent_img, pred_depth, poses[frame_id])
            imgs_to_visualize["warped_id_{}".format(frame_id)] = warped_adjacent_img

        return imgs_to_visualize
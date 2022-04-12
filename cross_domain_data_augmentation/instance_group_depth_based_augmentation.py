import copy
import math
import os
import random

import PIL
import PIL.Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import torch
from torch.utils.data import DataLoader

import camera_models
import dataloaders
import models
from cfg.config_dataset import get_cfg_dataset_defaults
from cfg.config_training import create_configuration
from dataloaders.synthia.dataset_single_instances_and_class_groups_synthia import SynthiaInstancesDataset
from io_utils import io_utils
from utils.plotting_like_cityscapes_utils import visu_depth_prediction as vdp
from utils.plotting_like_cityscapes_utils import CITYSCAPES_ID_TO_NAME_16
from torchvision import transforms

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

path = \
    r'.\cfg\yaml_files\train\depth_unsupervised\train_depth_cityscapes.yaml'
cfg = create_configuration(path)

# dataset 1
dcfg1 = cfg.datasets.configs[2]
dataset1 = dataloaders.get_dataset(dcfg1.dataset.name,
                                   'train',
                                   dcfg1.dataset.split,
                                   dcfg1)
loader1 = DataLoader(dataset1,
                     batch_size=1,
                     shuffle=True,
                     num_workers=0,
                     pin_memory=True,
                     drop_last=False)

normalized_camera_model1 = \
    camera_models.get_camera_model_from_file(dcfg1.dataset.camera,
                                             os.path.join(dcfg1.dataset.path,
                                                          "calib", "calib.txt"))

img_w1 = dcfg1.dataset.feed_img_size[0]
img_h1 = dcfg1.dataset.feed_img_size[1]

model_guda = models.get_model(cfg.model.type, cfg)
model_guda = model_guda.to('cuda:0')
path_to_model = r'D:\Depth-Semantic-UDA\experiments\guda'
model_file_name = r"checkpoint_epoch_113.pth"
checkpoint = io_utils.IOHandler.load_checkpoint(path_to_model, model_file_name, 0)
# Load pretrained weights for the model and the optimizer status
io_utils.IOHandler.load_weights(checkpoint, model_guda.get_networks(), None)
model_guda.eval()

# synthia instance loader
path_16_synthia = r'.\cfg\yaml_files\dataloaders\synthia_instances.yaml'
cfg_16_synthia = get_cfg_dataset_defaults()
cfg_16_synthia.merge_from_file(path_16_synthia)
cfg_16_synthia.freeze()

dataset_synthia = SynthiaInstancesDataset('train', None, cfg_16_synthia)
loader_synthia = DataLoader(dataset_synthia, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

if cfg.datasets.configs[1].dataset.num_classes != cfg_16_synthia.dataset.num_classes:
    raise ValueError('Both datasets do not contain the same amount of classes!')

for batch_idx, data in enumerate(loader1):
    print(f'################################ Image {batch_idx} ################################')
    for key, val in data.items():
        data[key] = val.to('cuda:0')

    prediction = model_guda.forward(data, dataset_id=2, predict_depth=True, train=False)[0]
    depth_pred1 = prediction['depth'][0][('depth', 0)].detach()
    raw_sigmoid = prediction['depth'][1][('disp', 0)].detach()

    fig, axs = plt.subplots(2, 5)

    # rgb images
    raw_rgb1 = data[('rgb', 0)][0].cpu() + torch.tensor([[[0.28689554, 0.32513303, 0.28389177]]]).transpose(0, 2)
    plt_rgb1 = raw_rgb1.numpy().transpose(1, 2, 0)
    axs[1, 0].imshow(plt_rgb1)

    # depth images
    axs[1, 1].imshow(vdp(1 / depth_pred1))
    average_bottom_depth = math.inf

    depth_between_instances = 1.0

    # -------------------------augmented target image---------------------------------
    for instance_idx, inst in enumerate(loader_synthia):
        #fig, axs = plt.subplots(2, 5)
        #axs[1, 0].imshow(plt_rgb1)
        #axs[1, 1].imshow(vdp(1 / depth_pred1))
        print(f'Instance Id: {instance_idx}')
        if instance_idx >= 10:
            break
        instance_rgb = inst['rgb'][0]
        instance_depth = inst['depth_dense'][0].squeeze(0)
        instance_semantic = inst['semantic'][0]
        instance_mask = torch.clone(instance_semantic)
        instance_mask[instance_mask == 250] = 0
        instance_mask = instance_mask.bool()

        class_id = instance_semantic[instance_mask][0].item()
        print(f'Object of class {CITYSCAPES_ID_TO_NAME_16[class_id]}')

        img_w_instance = instance_rgb.shape[2]
        img_h_instance = instance_rgb.shape[1]
        print(f'Instance Shape: [{img_h_instance}, {img_w_instance}]')

        number_pixels_bottom = torch.sum(instance_mask[-1, :].int())
        new_average_bottom_depth = torch.mean(instance_depth[instance_mask][-number_pixels_bottom:]).item()
        if np.abs(average_bottom_depth - new_average_bottom_depth) < depth_between_instances:
            continue
        average_bottom_depth = new_average_bottom_depth
        print(f'Depth at lower pixels of instance: {average_bottom_depth}')

        # some plotting
        raw_rgb0 = instance_rgb.cpu()
        plt_rgb0 = raw_rgb0.numpy().transpose(1, 2, 0)

        axs[0, 0].imshow(plt_rgb0)

        # find point where to insert the source instance
        axs[1, 1].imshow(raw_sigmoid.squeeze(0).squeeze(0).cpu())
        axs[0, 4].imshow(depth_pred1.squeeze(0).squeeze(0).cpu())
        depth_mask = (torch.abs(depth_pred1 - average_bottom_depth) < 0.2).squeeze(0).squeeze(0)
        best_row = torch.argmax(torch.sum(depth_mask.int(), dim=1)).item()
        best_row_logits = torch.zeros_like(depth_mask).to(torch.bool)
        best_row_logits[best_row, :] = True
        best_suited_depth = torch.logical_and(depth_mask, best_row_logits)

        axs[1, 2].imshow(depth_mask.cpu().numpy())
        axs[0, 2].imshow(best_row_logits.cpu().numpy())
        axs[0, 3].imshow(best_suited_depth.cpu())


        # gets all possible positions in the input image and select a random position
        position = (best_suited_depth == True).nonzero()  # 2d tensor containing h and w coordinate of true values
        if len(position) == 0:
            continue
        position = position[numpy.random.randint(0, position.shape[0])]  # select a random position
        print(f'Lower right corner of insertion {position}')
        # calculate padding values
        top = position[0] - img_h_instance
        left = position[1] - img_w_instance
        right = img_w1 - position[1]
        bottom = img_h1 - position[0]

        padded_mask = torch.clone(instance_mask)

        plt_now = True
        if top < 0:
            print(f'Inserted Object cut off at the top border {top}')
            instance_mask[0:-top, :] = False
            padded_mask = padded_mask[-top:, :]
            print(f'Mask shape: {padded_mask.shape}')
            top = 0
        if left < 0:
            print(f'Inserted Object cut off at the left border {left}')
            instance_mask[:, 0:-left] = False
            padded_mask = padded_mask[:, -left:]
            print(f'Mask shape: {padded_mask.shape}')
            left = 0

        padding = (left, top, right, bottom)
        print(padding)
        padder = transforms.Pad(padding, padding_mode='constant', fill=False)

        padded_mask = padder(padded_mask)
        print(f'Mask shape: {padded_mask.shape}')

        depth1_vals = depth_pred1[0][:, padded_mask]
        depth_mask = (instance_depth[instance_mask].to('cuda:0') < (depth1_vals + depth_between_instances)).cpu()
        instance_mask[instance_mask == True] = depth_mask  # cut away occluded part
        padded_mask[padded_mask == True] = depth_mask

        axs[0, 1].imshow(instance_mask)

        # insert into image

        raw_rgb1[:, padded_mask] = raw_rgb0[:, instance_mask]
        depth_pred1[0, 0, padded_mask] = instance_depth[instance_mask].to('cuda:0')
        axs[1, 3].imshow(raw_rgb1.cpu().numpy().transpose(1, 2, 0))
        axs[1, 4].imshow(vdp(1 / depth_pred1))

    if True:
        plt.show()
    else:
        plt.close(fig)
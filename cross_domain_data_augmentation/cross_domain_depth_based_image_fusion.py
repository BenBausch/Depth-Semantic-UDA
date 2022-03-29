import copy
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import camera_models
import dataloaders
import models
from cfg.config_training import create_configuration
from io_utils import io_utils
from utils.plotting_like_cityscapes_utils import visu_depth_prediction as vdp
from torchvision import transforms

# torch.manual_seed(0)
# np.random.seed(0)
# random.seed(0)

path = \
    r'/cfg/yaml_files/train/guda/train_guda_synthia_cityscapes.yaml'
cfg = create_configuration(path)
# dataset 0
dcfg0 = cfg.datasets.configs[0]
dataset0 = dataloaders.get_dataset(dcfg0.dataset.name,
                                   'train',
                                   dcfg0.dataset.split,
                                   dcfg0)
loader0 = DataLoader(dataset0,
                     batch_size=1,
                     shuffle=True,
                     num_workers=0,
                     pin_memory=True,
                     drop_last=False)
normalized_camera_model0 = \
    camera_models.get_camera_model_from_file(dcfg0.dataset.camera,
                                             os.path.join(dcfg0.dataset.path,
                                                          "calib", "calib.txt"))
img_w0 = dcfg0.dataset.feed_img_size[0]
img_h0 = dcfg0.dataset.feed_img_size[1]

# dataset 1
dcfg1 = cfg.datasets.configs[1]
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
model_file_name = r"checkpoint_epoch_43.pth"
checkpoint = io_utils.IOHandler.load_checkpoint(path_to_model, model_file_name, 0)
# Load pretrained weights for the model and the optimizer status
io_utils.IOHandler.load_weights(checkpoint, model_guda.get_networks(), None)
model_guda.eval()

for batch_idx, data in enumerate(zip(loader0, loader1)):
    for key, val in data[0].items():
        data[0][key] = val.to('cuda:0')
    for key, val in data[1].items():
        data[1][key] = val.to('cuda:0')

    # filter out virtual scenes that are not from a driving perspective
    # calculate average depth of bottom two pixel rows
    print(data[0]['depth_dense'][0].shape)
    avg_depth = torch.mean(data[0]['depth_dense'][0][:, -2:, :])
    if avg_depth > 3.0:  # greater than a meter distance to camera
        print(f'not a valid scene with average bottom depth of {avg_depth}')
        continue
    else:
        print(f'VALID SCENE WITH DEPTH OF {avg_depth}')

    prediction = model_guda.forward(data)
    depth_pred1 = prediction[1]['depth'][0][('depth', 0)].detach()

    fig, axs = plt.subplots(2, 3)

    # rgb images
    raw_rgb0 = data[0][('rgb', 0)][0].cpu() + torch.tensor([[[0.314747602, 0.277402550, 0.248091921]]]).transpose(0, 2)
    plt_rgb0 = raw_rgb0.numpy().transpose(1, 2, 0)
    raw_rgb1 = data[1][('rgb', 0)][0].cpu() + torch.tensor([[[0.28689554, 0.32513303, 0.28389177]]]).transpose(0, 2)
    plt_rgb1 = raw_rgb1.numpy().transpose(1, 2, 0)
    axs[0, 0].imshow(plt_rgb0)
    axs[1, 0].imshow(plt_rgb1)

    # depth images
    axs[0, 1].imshow(vdp(1 / data[0]['depth_dense'][0]))
    axs[1, 1].imshow(vdp(1 / depth_pred1))

    # -------------------------augmented target image---------------------------------
    # assumes target image bigger than source image!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    mask = data[0]['semantic'][0] == 6

    raw_rgb0[:, ~mask] = 0.0
    axs[0, 2].imshow(raw_rgb0.cpu().numpy().transpose(1, 2, 0))
    depth0_vals = data[0]['depth_dense'][0][:, mask]
    print(depth0_vals.shape)

    # calculate padding
    pad_left = random.randint(0, int(img_w1 - img_w0))
    pad_right = int(img_w1 - img_w0) - pad_left
    pad_top = random.randint(0, int(img_h1 - img_h0))
    pad_bottom = int(img_h1 - img_h0) - pad_top
    padding = (pad_left, pad_top, pad_right, pad_bottom)
    padder = transforms.Pad(padding, padding_mode='constant', fill=False)

    # attention to depth within mask
    mask1 = padder(mask)
    depth1_vals = depth_pred1[0][:, mask1]
    print(depth1_vals.shape)
    depth_mask = depth0_vals < (depth1_vals + 2.0)
    mask1[mask1 == True] = depth_mask
    mask[mask == True] = depth_mask
    mask1 = mask1.reshape((img_h1, img_w1))


    # insert object into target image
    raw_rgb1[:, mask1] = raw_rgb0[:, mask]
    axs[1, 2].imshow(raw_rgb1.cpu().numpy().transpose(1, 2, 0))

    plt.show()

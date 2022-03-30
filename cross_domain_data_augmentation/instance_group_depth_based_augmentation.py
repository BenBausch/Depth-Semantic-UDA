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
from torchvision import transforms

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

path = \
    r'./cfg/yaml_files/train/guda/train_guda_synthia_cityscapes.yaml'
cfg = create_configuration(path)

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

r"""# load a single instance to try out algorithm
path_to_instance = r'C:\Users\benba\Documents\University\Masterarbeit\data\single_class_instances'
instance_file_name = r'person\14.png'
instance_rgb = os.path.join(path_to_instance, 'RGB', instance_file_name)
instance_depth = os.path.join(path_to_instance, 'DEPTH', instance_file_name)
instance_mask = os.path.join(path_to_instance, 'MASK', instance_file_name)

instance_rgb = PIL.Image.open(instance_rgb).convert('RGB')
instance_depth = cv2.imread(instance_depth, cv2.IMREAD_UNCHANGED).astype(np.int32)
instance_mask = cv2.imread(instance_mask, cv2.IMREAD_UNCHANGED)

img_w_instance_bevor_resize = instance_rgb.size[1]
img_h_instance_bevor_resize = instance_rgb.size[0]

instance_rgb = instance_rgb.resize([int(img_h_instance_bevor_resize/4), int(img_w_instance_bevor_resize/4)],
                                   resample=PIL.Image.BICUBIC)
instance_depth = cv2.resize(instance_depth, [int(img_h_instance_bevor_resize/4), int(img_w_instance_bevor_resize/4)],
                            interpolation=cv2.INTER_NEAREST)
instance_mask = cv2.resize(instance_mask, [int(img_h_instance_bevor_resize/4), int(img_w_instance_bevor_resize/4)],
                           interpolation=cv2.INTER_NEAREST)

instance_mask[instance_mask == 250] = 0

instance_rgb = transforms.ToTensor()(instance_rgb).double()
instance_depth = torch.DoubleTensor(instance_depth) / 100
instance_mask = torch.BoolTensor(instance_mask.astype(bool))

img_w_instance = instance_rgb.shape[2]
img_h_instance = instance_rgb.shape[1]

# get the average depth at the feet
number_pixels_bottom = torch.sum(instance_mask[-1, :].int())
average_bottom_depth = torch.mean(instance_depth[instance_mask][-number_pixels_bottom:]).item()
print(f'Average depth at feet: {average_bottom_depth}')

if True:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(instance_rgb.numpy().transpose(1, 2, 0))
    ax2.imshow(instance_mask)
    ax3.imshow(vdp(1/instance_depth.unsqueeze(0)))
    plt.show()"""

# synthia instance laoder
path_16_synthia = r'.\cfg\yaml_files\dataloaders\synthia_instances.yaml'
cfg_16_synthia = get_cfg_dataset_defaults()
cfg_16_synthia.merge_from_file(path_16_synthia)
cfg_16_synthia.freeze()

dataset_synthia = SynthiaInstancesDataset('train', None, cfg_16_synthia)
loader_synthia = DataLoader(dataset_synthia, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

if cfg.datasets.configs[1].dataset.num_classes != cfg_16_synthia.dataset.num_classes:
    raise ValueError('Both datasets do not contain the same amount of classes!')

for batch_idx, data in enumerate(loader1):
    for key, val in data.items():
        data[key] = val.to('cuda:0')

    prediction = model_guda.forward(data, dataset_id=1, predict_depth=True, train=False)[0]
    depth_pred1 = prediction['depth'][0][('depth', 0)].detach()

    fig, axs = plt.subplots(2, 5)

    # rgb images
    raw_rgb1 = data[('rgb', 0)][0].cpu() + torch.tensor([[[0.28689554, 0.32513303, 0.28389177]]]).transpose(0, 2)
    plt_rgb1 = raw_rgb1.numpy().transpose(1, 2, 0)
    axs[1, 0].imshow(plt_rgb1)

    # depth images
    axs[1, 1].imshow(vdp(1 / depth_pred1))
    #average_bottom_depth = math.inf
    # -------------------------augmented target image---------------------------------
    for instance_idx, inst in enumerate(loader_synthia):
        print(instance_idx)
        if instance_idx >= 3:
            break
        instance_rgb = inst['rgb'][0]
        instance_depth = inst['depth_dense'][0].squeeze(0)
        instance_semantic = inst['semantic'][0]
        instance_mask = torch.clone(instance_semantic)
        instance_mask[instance_mask == 250] = 0
        instance_mask = instance_mask.bool()

        img_w_instance = instance_rgb.shape[2]
        img_h_instance = instance_rgb.shape[1]
        print(img_h_instance)
        print(img_w_instance)

        number_pixels_bottom = torch.sum(instance_mask[-1, :].int())
        #if average_bottom_depth < torch.mean(instance_depth[instance_mask][-number_pixels_bottom:]).item():
        #    continue
        average_bottom_depth = torch.mean(instance_depth[instance_mask][-number_pixels_bottom:]).item()

        # some plotting
        raw_rgb0 = instance_rgb.cpu()
        plt_rgb0 = raw_rgb0.numpy().transpose(1, 2, 0)

        axs[0, 0].imshow(plt_rgb0)
        axs[0, 1].imshow(vdp(1 / instance_depth.unsqueeze(0)))

        # find point where to insert the source instance
        depth_mask = (torch.abs(depth_pred1 - average_bottom_depth) < 0.2).squeeze(0).squeeze(0)
        best_row = torch.argmax(torch.sum(depth_mask.int(), dim=1)).item()
        best_row_logits = torch.zeros_like(depth_mask).to(torch.bool)
        best_row_logits[best_row, :] = True
        best_suited_depth = torch.logical_and(depth_mask, best_row_logits)

        axs[1, 2].imshow(depth_mask.cpu().numpy())
        axs[0, 2].imshow(best_row_logits.cpu().numpy())
        axs[0, 3].imshow(best_suited_depth.cpu())


        #  gets all possible positions in the input image and select a random position
        position = (best_suited_depth == True).nonzero()  # 2d tensor containing h and w coordinate of true values
        position = position[numpy.random.randint(0, position.shape[0])]  # select a random position
        print(position)
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
        depth_mask = (instance_depth[instance_mask].to('cuda:0') < (depth1_vals + 2.0)).cpu()
        instance_mask[instance_mask == True] = depth_mask  # cut away occluded part
        padded_mask[padded_mask == True] = depth_mask

        # insert into image
        # axs[1, 3].imshow(padder(instance_mask))
        print(instance_mask.shape)
        print(raw_rgb0.shape)
        raw_rgb1[:, padded_mask] = raw_rgb0[:, instance_mask]
        depth_pred1[0, 0, padded_mask] = instance_depth[instance_mask].to('cuda:0')
        axs[1, 3].imshow(raw_rgb1.cpu().numpy().transpose(1, 2, 0))
        axs[1, 4].imshow(vdp(1 / depth_pred1))

    if plt_now:
        plt.show()
    else:
        plt.close(fig)
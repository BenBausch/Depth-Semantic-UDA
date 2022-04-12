import copy
import os
import random

import PIL.Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import camera_models
import dataloaders
import models
from cfg.config_dataset import get_cfg_dataset_defaults
import torchvision.transforms.functional as F
import torchvision.transforms
from cfg.config_training import create_configuration
from io_utils import io_utils
from utils.plotting_like_cityscapes_utils import visu_depth_prediction as vdp
from utils.plotting_like_cityscapes_utils import CITYSCAPES_ID_TO_NAME_16
from torchvision import transforms
from utils.constans import IGNORE_INDEX_SEMANTIC, IGNORE_VALUE_DEPTH

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

path_16 = r'./cfg/yaml_files/dataloaders/synthia.yaml'
cfg_16 = get_cfg_dataset_defaults()
cfg_16.merge_from_file(path_16)
cfg_16.freeze()

# dataset 0
dcfg0 = cfg_16
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

basepath = r'..\data\single_class_instances'

semantic_counter = 0

#path_file = r'C:\Users\benba\Documents\University\Masterarbeit\data\single_instances\DEPTH\motorcycle\1.png'
#d = cv2.imread(path_file, cv2.IMREAD_UNCHANGED).astype(np.int32)
#print(d)
#d = torch.FloatTensor(d) / 100
#print(d)
#plt.imshow(vdp(1/d.unsqueeze(0)))
#plt.show()


for batch_idx, data in enumerate(loader0):
    for key, val in data.items():
        data[key] = val.to('cuda:0')

    # filter out virtual scenes that are not from a driving perspective
    # calculate average depth of bottom two pixel rows
    avg_depth = torch.mean(data['depth_dense'][0][:, -2:, :])
    if avg_depth > 3.0:  # greater than a meter distance to camera
        print(f'not a valid scene with average bottom depth of {avg_depth}')
        continue
    else:
        print(f'VALID SCENE WITH DEPTH OF {avg_depth}')

    semantic_ids = torch.unique(data['semantic'][0])[1:]  # remove id 0 of backgound
    stuff_or_dynamic_classes_to_be_ignored = [0, 1, 8, 9, 10, 11, 12, 13, 14, 15, 250]

    for _, inst in enumerate(semantic_ids):
        if inst in stuff_or_dynamic_classes_to_be_ignored:
            print('skipped')
            continue
        else:
            print('accepted')
        print(f'semantic with id {inst}')

        # rule 1:
        size_of_instance = torch.sum((data['semantic'][0] == inst).long())
        print(f'Consists of {size_of_instance} pixels!')
        if size_of_instance < 5000:
            continue

        # rule 2: ignore semantics where large portion is cut off at the border
        mask = data['semantic'][0] == inst
        sum_of_true_vals_at_border = torch.LongTensor([0]).to('cuda:0')
        sum_of_true_vals_at_border += torch.sum((data['semantic'][0][0, :] == inst).long())  # left
        sum_of_true_vals_at_border += torch.sum((data['semantic'][0][-1, :] == inst).long())  # right
        sum_of_true_vals_at_border += torch.sum((data['semantic'][0][:, 0] == inst).long())  # top
        sum_of_true_vals_at_border += torch.sum((data['semantic'][0][:, -1] == inst).long())  # bottom
        if sum_of_true_vals_at_border > 0:
            continue

        # get min max of x and y coordinates of true vals in mask
        print(mask.nonzero().shape)
        min_x = torch.min(mask.nonzero()[:, 1])
        print(f'left: {min_x}')
        max_x = torch.max(mask.nonzero()[:, 1])
        print(f'right: {max_x}')
        min_y = torch.min(mask.nonzero()[:, 0])
        print(f'top: {min_y}')
        max_y = torch.max(mask.nonzero()[:, 0])
        print(f'bottom: {max_y}')

        #fig, axs = plt.subplots(2, 4)

        # rgb images
        raw_rgb0 = data[('rgb', 0)][0].cpu() + torch.tensor([[[0.314747602, 0.277402550, 0.248091921]]]).transpose(0, 2)

        #plt_rgb0 = raw_rgb0.numpy().transpose(1, 2, 0)
        #axs[0, 0].imshow(plt_rgb0)

        # depth images
        #axs[0, 1].imshow(vdp(1 / data['depth_dense'][0]))
        #axs[0, 3].imshow(data['semantic'][0].cpu().numpy())

        # -------------------------crop data---------------------------------
        # assumes target image bigger than source image!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        raw_rgb0[:, ~mask] = 0.0
        #axs[0, 2].imshow(raw_rgb0.cpu().numpy().transpose(1, 2, 0))

        cropped_to_content_rgb = F.crop(raw_rgb0, top=min_y, left=min_x, width=max_x - min_x, height=max_y - min_y)
        #axs[1, 0].imshow(cropped_to_content_rgb.numpy().transpose(1, 2, 0))

        semantic_mask = (data['semantic'][0] == inst).long()
        semantic_mask[semantic_mask == 0] = IGNORE_INDEX_SEMANTIC
        semantic_mask[semantic_mask == 1] = inst
        cropped_to_content_semantic = F.crop(semantic_mask, top=min_y, left=min_x, width=max_x - min_x, height=max_y - min_y)
        #axs[1, 1].imshow(cropped_to_content_semantic.cpu().numpy())

        #data['depth_dense'][0][:, ~mask] = IGNORE_VALUE_DEPTH  # assign max depth to rest of image
        #axs[1, 2].imshow(data['depth_dense'][0].cpu().numpy().transpose(1, 2, 0))
        cropped_to_content_depth = F.crop(data['depth_dense'][0], top=min_y, left=min_x, width=max_x - min_x, height=max_y - min_y)
        cropped_to_content_depth = (cropped_to_content_depth * 100)
        #axs[1, 2].imshow(vdp(1 / cropped_to_content_depth))

        # --------------------- save the cropped semantic and all labels
        cityscapes_class_name = CITYSCAPES_ID_TO_NAME_16[data['semantic'][0][mask][0].item()]

        if not os.path.exists(os.path.join(basepath, 'RGB', cityscapes_class_name)):
            os.mkdir(os.path.join(basepath, 'RGB', cityscapes_class_name))
        if not os.path.exists(os.path.join(basepath, 'DEPTH', cityscapes_class_name)):
            os.mkdir(os.path.join(basepath, 'DEPTH', cityscapes_class_name))
        if not os.path.exists(os.path.join(basepath, 'MASK', cityscapes_class_name)):
            os.mkdir(os.path.join(basepath, 'MASK', cityscapes_class_name))

        print(f'semantic of class {cityscapes_class_name}')

        to_pil = transforms.ToPILImage()

        #plt.show()

        if True:
            cropped_to_content_rgb = to_pil(cropped_to_content_rgb)
            cropped_to_content_rgb.save(os.path.join(basepath, 'RGB',
                                                     cityscapes_class_name, f'{str(semantic_counter)}.png'))

            cv2.imwrite(os.path.join(basepath, 'MASK', cityscapes_class_name, f'{str(semantic_counter)}.png'),
                        cropped_to_content_semantic.cpu().numpy().astype(np.int64),
                        [cv2.IMREAD_ANYDEPTH])

            cv2.imwrite(os.path.join(basepath, 'DEPTH', cityscapes_class_name, f'{str(semantic_counter)}.png'),
                        cropped_to_content_depth.cpu().numpy().transpose(1, 2, 0).astype(np.uint16),
                        [cv2.IMREAD_ANYDEPTH])

        semantic_counter += 1


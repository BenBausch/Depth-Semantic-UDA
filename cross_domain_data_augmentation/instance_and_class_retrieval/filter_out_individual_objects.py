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
from utils import IGNORE_INDEX_SEMANTIC, IGNORE_VALUE_DEPTH
from utils.plotting_like_cityscapes_utils import visu_depth_prediction as vdp
from utils.plotting_like_cityscapes_utils import CITYSCAPES_ID_TO_NAME_16
from torchvision import transforms

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

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

instance_counter = 0

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

    instance_ids = torch.unique(data['instance'][0])[1:]  # remove id 0 of backgound

    for _, inst in enumerate(instance_ids):

        # rules for instance selection!
        # rule 1: size of the instance
        size_of_instance = torch.sum((data['instance'][0] == inst).long())
        if size_of_instance < 14592:
            continue

        # rule 2: ignore instances where large portion is cut off at the border
        mask = data['instance'][0] == inst
        sum_of_true_vals_at_border = torch.LongTensor([0]).to('cuda:0')
        sum_of_true_vals_at_border += torch.sum((data['instance'][0][0, :] == inst).long())  # left
        sum_of_true_vals_at_border += torch.sum((data['instance'][0][-1, :] == inst).long())  # right
        sum_of_true_vals_at_border += torch.sum((data['instance'][0][:, 0] == inst).long())  # top
        sum_of_true_vals_at_border += torch.sum((data['instance'][0][:, -1] == inst).long())  # bottom
        if sum_of_true_vals_at_border > 0:
            continue

        # select
        # pad with false since the borders are false anyway due to rule 2
        # border inside object
        padder_h = torch.nn.ConstantPad2d((0, 0, 0, 1), False)  # bottom padding with false
        padder_w = torch.nn.ConstantPad2d((0, 1, 0, 0), False)  # right padding with false
        inv_mask = ~mask
        border_mask = inv_mask[0:, :] == padder_h(mask[1:, :])
        border_mask = torch.logical_or(border_mask, inv_mask[:, 0:] == padder_w(mask[:, 1:]))

        # get min max of x and y coordinates of true vals in mask
        print(mask.nonzero().shape)
        min_x = torch.min(mask.nonzero()[:, 1])
        max_x = torch.max(mask.nonzero()[:, 1])
        min_y = torch.min(mask.nonzero()[:, 0])
        max_y = torch.max(mask.nonzero()[:, 0])

        #fig, axs = plt.subplots(2, 4)

        # rgb images
        raw_rgb0 = data[('rgb', 0)][0].cpu() + torch.tensor([[[0.314747602, 0.277402550, 0.248091921]]]).transpose(0, 2)

        #plt_rgb0 = raw_rgb0.numpy().transpose(1, 2, 0)
        #axs[0, 0].imshow(plt_rgb0)

        # depth images
        #axs[0, 1].imshow(vdp(1 / data['depth_dense'][0]))
        #axs[0, 3].imshow(data['instance'][0].cpu().numpy())
        #axs[1, 3].imshow(border_mask.cpu())

        # -------------------------crop data---------------------------------
        # rgb
        raw_rgb0[:, ~mask] = 0.0
        #axs[0, 2].imshow(raw_rgb0.cpu().numpy().transpose(1, 2, 0))

        cropped_to_content_rgb = F.crop(raw_rgb0, top=min_y, left=min_x, width=max_x - min_x, height=max_y - min_y)
        #axs[1, 0].imshow(cropped_to_content_rgb.numpy().transpose(1, 2, 0))

        # semantic
        semantic_mask = (data['instance'][0] == inst).long()
        semantic_mask[semantic_mask == 0] = IGNORE_INDEX_SEMANTIC
        semantic_class_id = data['semantic'][:, data['instance'][0] == inst][0]  # todo check this
        semantic_mask[semantic_mask == 1] = semantic_class_id  # label asign label
        cropped_to_content_semantic = F.crop(semantic_mask, top=min_y, left=min_x, width=max_x - min_x,
                                             height=max_y - min_y)
        #axs[1, 1].imshow(semantic_mask.cpu().numpy())

        # depth
        cropped_to_content_depth = F.crop(data['depth_dense'][0], top=min_y, left=min_x, width=max_x - min_x,
                                          height=max_y - min_y)
        cropped_to_content_depth = (cropped_to_content_depth * 100)
        #axs[1, 2].imshow(vdp(1 / cropped_to_content_depth))

        # rule 3
        # calculate avg depth outside mask for each instance/non-dynamic class except for road, sky, sidewalk pixels
        # compare to average depth of instance if smaller object not occluded

        cropped_outside_instance_mask = data['semantic'][0] != 0  # not road
        print('dewddd')
        print(cropped_outside_instance_mask.shape)
        print(f'Unique class ids: {torch.unique(semantic_class_id)}')
        cropped_outside_instance_mask = torch.logical_and(cropped_outside_instance_mask,
                                                          data['semantic'][0] != 1)  # not sidewalk
        cropped_outside_instance_mask = torch.logical_and(cropped_outside_instance_mask,
                                                          data['semantic'][0] != 9)  # not sky
        cropped_outside_instance_mask = torch.logical_xor(cropped_outside_instance_mask,
                                                          data['instance'][0] == inst)  # not the instance itself

        cropped_instance_mask = F.crop(data['instance'][0] == inst, top=min_y, left=min_x,
                                       width=max_x - min_x,
                                       height=max_y - min_y)

        cropped_outside_instance_mask = F.crop(cropped_outside_instance_mask, top=min_y, left=min_x,
                                               width=max_x - min_x,
                                               height=max_y - min_y)

        cropped_instance_data = F.crop(data['instance'][0], top=min_y, left=min_x,
                                       width=max_x - min_x,
                                       height=max_y - min_y)

        cropped_class_data = F.crop(data['semantic'][0], top=min_y, left=min_x,
                                    width=max_x - min_x,
                                    height=max_y - min_y)

        pos_occluding_instances = torch.unique(cropped_instance_data[cropped_outside_instance_mask])[1:]
        pos_occluding_classes = torch.unique(cropped_class_data[cropped_outside_instance_mask])[1:]
        print(pos_occluding_classes)
        print(pos_occluding_instances)

        avg_depth_inst = torch.mean(cropped_to_content_depth[:, cropped_instance_mask])

        inst_bbox_size = cropped_instance_mask.shape[0] * cropped_instance_mask.shape[1]

        print('##############################################################################################')

        print(f'Depth of instance of interest: {avg_depth_inst}')
        
        reject_this_instance = False

        for j, poi in enumerate(pos_occluding_instances):
            cropped_poi_mask = F.crop(data['instance'][0] == poi, top=min_y, left=min_x,
                                      width=max_x - min_x, height=max_y - min_y)
            if torch.sum(cropped_poi_mask.long()) < 0.01 * inst_bbox_size:  # occlusions of at least 100 pixels
                continue

            avg_depth_poi = torch.mean(cropped_to_content_depth[:, cropped_poi_mask])
            print(f'Average deph of {poi} is {avg_depth_poi}')

            if avg_depth_inst > avg_depth_poi:
                print('Depth is smaller')
                reject_this_instance = True
                break
            print(f'Depth of poi instance {poi}: {avg_depth_poi}')

        if reject_this_instance:
            #plt.close(fig)
            continue

        for j, poc in enumerate(pos_occluding_classes):
            cropped_poc_mask = F.crop(data['semantic'][0] == poc, top=min_y, left=min_x,
                                      width=max_x - min_x, height=max_y - min_y)
            if torch.sum(cropped_poc_mask.long()) < 0.01 * inst_bbox_size:  # occlusions of at least 100 pixels
                continue

            avg_depth_poc = torch.mean(cropped_to_content_depth[:, cropped_poc_mask])
            print(f'Average deph of {poc} is {avg_depth_poc}')

            if avg_depth_inst > avg_depth_poc:
                print('Depth is smaller')
                reject_this_instance = True
                break
            print(f'Depth of poc instance {poc}: {avg_depth_poc}')

        if reject_this_instance:
            #plt.close(fig)
            continue

        #axs[1, 3].imshow(cropped_outside_instance_mask.cpu())

        print(f'Instance with id {inst}')
        print(f'Consists of {size_of_instance} pixels!')
        print(f'bottom: {max_y}')
        print(f'top: {min_y}')
        print(f'right: {max_x}')
        print(f'left: {min_x}')

        # --------------------- save the cropped instance and all labels
        cityscapes_class_name = CITYSCAPES_ID_TO_NAME_16[data['semantic'][0][mask][0].item()]

        if not os.path.exists(os.path.join(basepath, 'RGB', cityscapes_class_name)):
            os.mkdir(os.path.join(basepath, 'RGB', cityscapes_class_name))
        if not os.path.exists(os.path.join(basepath, 'DEPTH', cityscapes_class_name)):
            os.mkdir(os.path.join(basepath, 'DEPTH', cityscapes_class_name))
        if not os.path.exists(os.path.join(basepath, 'MASK', cityscapes_class_name)):
            os.mkdir(os.path.join(basepath, 'MASK', cityscapes_class_name))

        print(f'Instance of class {cityscapes_class_name}')

        to_pil = transforms.ToPILImage()

        #plt.show()

        if True:
            cropped_to_content_rgb = to_pil(cropped_to_content_rgb)
            cropped_to_content_rgb.save(os.path.join(basepath, 'RGB',
                                                     cityscapes_class_name, f'{str(instance_counter)}.png'))

            cv2.imwrite(os.path.join(basepath, 'MASK', cityscapes_class_name, f'{str(instance_counter)}.png'),
                        cropped_to_content_semantic.cpu().numpy().astype(np.int64),
                        [cv2.IMREAD_ANYDEPTH])

            cv2.imwrite(os.path.join(basepath, 'DEPTH', cityscapes_class_name, f'{str(instance_counter)}.png'),
                        cropped_to_content_depth.cpu().numpy().transpose(1, 2, 0).astype(np.uint16),
                        [cv2.IMREAD_ANYDEPTH])

        instance_counter += 1

        #plt.close(fig)

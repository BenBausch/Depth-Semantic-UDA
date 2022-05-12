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
from utils.plotting_like_cityscapes_utils import semantic_id_tensor_to_rgb_numpy_array as s2rgb
from utils.constans import IGNORE_INDEX_SEMANTIC
from utils.plotting_like_cityscapes_utils import CITYSCAPES_ID_TO_NAME_16
from torchvision import transforms


def fix_seeds():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


def get_configuration():
    # create configuration files
    path_to_model_yaml = os.path.join('.', 'cfg', 'yaml_files', 'data_augmentation',
                                      'generate_synthia_augmented_cityscapes.yaml')
    path_to_cityscapes_yaml = os.path.join('.', 'cfg', 'yaml_files', 'data_augmentation',
                                           'cityscapes_semantic.yaml')
    path_to_synthia_yaml = os.path.join('.', 'cfg', 'yaml_files', 'data_augmentation',
                                        'synthia_instances.yaml')
    cfg_model = create_configuration(path_to_model_yaml)
    cfg_cityscapes = get_cfg_dataset_defaults()
    cfg_cityscapes.merge_from_file(path_to_cityscapes_yaml)
    cfg_cityscapes.freeze()
    cfg_synthia = get_cfg_dataset_defaults()
    cfg_synthia.merge_from_file(path_to_synthia_yaml)
    cfg_synthia.freeze()
    return cfg_model, cfg_synthia, cfg_cityscapes


def create_model(cfg_model):
    model = models.get_model(cfg_model.model.type, cfg_model)
    model = model.to('cuda:0')
    checkpoint = io_utils.IOHandler.load_checkpoint(cfg_model.checkpoint.path_base, cfg_model.checkpoint.filename, 0)
    # Load pretrained weights for the model and the optimizer status
    io_utils.IOHandler.load_weights(checkpoint, model.get_networks(), None)
    model.eval()
    return model


def create_dataloaders(cfg_cityscapes, cfg_synthia):
    cityscapes_dataset = dataloaders.get_dataset(cfg_cityscapes.dataset.name,
                                                 'train',
                                                 cfg_cityscapes.dataset.split,
                                                 cfg_cityscapes)
    cityscapes_loader = DataLoader(cityscapes_dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=True,
                                   drop_last=False)

    synthia_dataset = SynthiaInstancesDataset('train', None, cfg_synthia)
    synthia_loader = DataLoader(synthia_dataset,
                                batch_size=1,
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True,
                                drop_last=False)

    if cfg_cityscapes.dataset.num_classes != cfg_synthia.dataset.num_classes:
        raise ValueError('Both datasets do not contain the same amount of classes!')

    return cityscapes_loader, synthia_loader


def get_insert_position(depth_cityscapes, average_bottom_depth_of_instance, img_h1, threshold=0.2):
    # find point where to insert the source instance
    depth_mask = (torch.abs(depth_cityscapes - average_bottom_depth_of_instance) < threshold).squeeze(0).squeeze(0)

    # check the best row to insert in the lower half of the target image
    best_row = torch.argmax(torch.sum(depth_mask.int(), dim=1)[int(img_h1/2):]).item() + int(img_h1/2)
    best_row_logits = torch.zeros_like(depth_mask).to(torch.bool)
    best_row_logits[best_row, :] = True
    best_suited_depth = torch.logical_and(depth_mask, best_row_logits)

    # gets all possible positions in the input image and select a random position
    position = (best_suited_depth == True).nonzero()  # 2d tensor containing h and w coordinate of true values
    if len(position) == 0:
        return None

    position = position[numpy.random.randint(0, position.shape[0])]  # select a random position

    return depth_mask, best_row_logits, best_suited_depth, position


def get_padding_values(position, img_h_instance, img_w_instance, img_w1, img_h1, instance_mask):
    top = position[0] - img_h_instance
    left = position[1] - img_w_instance
    right = img_w1 - position[1]
    bottom = img_h1 - position[0]

    padded_mask = torch.clone(instance_mask)

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
    return padding, padded_mask


def generate_synthia_augmented_cityscapes(cityscapes_loader, synthia_loader, model, img_h1, img_w1, base_path):
    number_instances_to_insert = 20
    depth_between_instances = 0.2

    to_pil = transforms.ToPILImage()

    for batch_idx, data in enumerate(cityscapes_loader):
        if batch_idx <= 126:
            continue
        print(f'################################ Image {batch_idx} ################################')
        for key, val in data.items():
            data[key] = val.to('cuda:0')

        prediction = model.forward(data, dataset_id=2, predict_depth=True, train=False)[0]
        depth_cityscapes = prediction['depth'][0][('depth', 0)].detach()
        non_modified_depth = depth_cityscapes.clone()

        raw_sigmoid = prediction['depth'][1][('disp', 0)].detach()

        #fig, axs = plt.subplots(2, 5)
        #fig_2, axs_2 = plt.subplots(1, 3)

        # rgb images
        rgb_cityscapes = data[('rgb', 0)][0].cpu()

        rgb_cityscapes_pil = to_pil(rgb_cityscapes)
        rgb_cityscapes_pil.save(os.path.join(base_path, 'RGB_UNAUG', f'{str(batch_idx)}.png'))

        plt_rgb1 = rgb_cityscapes.numpy().transpose(1, 2, 0)

        #axs[1, 0].imshow(plt_rgb1)
        #axs_2[0].imshow(plt_rgb1)

        # depth images
        #axs[1, 1].imshow(vdp(1 / depth_cityscapes))
        average_bottom_depth = math.inf

        succesful_inserts = 0

        joined_instance_masks = torch.zeros((rgb_cityscapes.shape[1], rgb_cityscapes.shape[2])).bool()

        new_semantic_mask = torch.zeros((img_h1, img_w1)) + IGNORE_INDEX_SEMANTIC

        # -------------------------augment target image---------------------------------
        for instance_idx, inst in enumerate(synthia_loader):

            print(f'Instance Id: {instance_idx}')

            if succesful_inserts >= number_instances_to_insert:
                break

            instance_rgb = inst['rgb'][0]
            instance_depth = inst['depth_dense'][0].squeeze(0)
            instance_semantic = inst['semantic'][0]
            instance_mask = torch.clone(instance_semantic)
            instance_mask[instance_mask == 250] = 0
            instance_mask = instance_mask.bool()
            semantic_label = instance_semantic[instance_mask][0].item()

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

            #  retrieve insert position
            try:
                depth_mask, best_row_logits, best_suited_depth, position = \
                    get_insert_position(non_modified_depth, average_bottom_depth, img_h1)
            except:
                continue
            print(f'Lower right corner of insertion {position}')

            # some plotting
            raw_rgb0 = instance_rgb.cpu()
            #plt_rgb0 = raw_rgb0.numpy().transpose(1, 2, 0)
            #axs[0, 0].imshow(plt_rgb0)
            #axs[1, 1].imshow(raw_sigmoid.squeeze(0).squeeze(0).cpu())
            #axs[0, 4].imshow(depth_cityscapes.squeeze(0).squeeze(0).cpu())
            #axs[1, 2].imshow(depth_mask.cpu().numpy())
            #axs[0, 2].imshow(best_row_logits.cpu().numpy())
            #axs[0, 3].imshow(best_suited_depth.cpu())

            # calculate padding values
            padding, padded_mask = get_padding_values(position, img_h_instance, img_w_instance, img_w1, img_h1,
                                                      instance_mask)
            print(f'Padding instance like {padding}')
            padder = transforms.Pad(padding, padding_mode='constant', fill=False)

            padded_mask = padder(padded_mask)
            print(f'Mask shape: {padded_mask.shape}')

            # prevent overlapping synthia instances
            if torch.logical_and(joined_instance_masks, padded_mask).any():
                continue
            # fix, axs = plt.subplots(1, 3)
            # axs[0].imshow(joined_instance_masks)
            # axs[1].imshow(padded_mask)
            joined_instance_masks = torch.logical_or(joined_instance_masks, padded_mask)
            # axs[2].imshow(joined_instance_masks)
            # plt.show()

            # cut away occlusions
            depth1_vals = depth_cityscapes[0][:, padded_mask]
            depth_mask = (instance_depth[instance_mask].to('cuda:0') < (depth1_vals + depth_between_instances)).cpu()
            instance_mask[instance_mask == True] = depth_mask  # cut away occluded part
            padded_mask[padded_mask == True] = depth_mask
            # axs[0, 1].imshow(instance_mask)

            # insert into image
            rgb_cityscapes[:, padded_mask] = raw_rgb0[:, instance_mask]
            depth_cityscapes[0, 0, padded_mask] = instance_depth[instance_mask].to('cuda:0')
            #axs[1, 3].imshow(rgb_cityscapes.cpu().numpy().transpose(1, 2, 0))
            #axs_2[1].imshow(rgb_cityscapes.cpu().numpy().transpose(1, 2, 0))
            #axs[1, 4].imshow(vdp(1 / depth_cityscapes))

            # create semantic mask
            new_semantic_mask[padded_mask] = semantic_label
            #axs_2[2].imshow(s2rgb(new_semantic_mask.unsqueeze(0), 16))

        rgb_cityscapes = to_pil(rgb_cityscapes)
        rgb_cityscapes.save(os.path.join(base_path, 'RGB', f'{str(batch_idx)}.png'))

        cv2.imwrite(os.path.join(base_path, 'LABELS', f'{str(batch_idx)}.png'),
                    new_semantic_mask.cpu().numpy().astype(np.int64),
                    [cv2.IMREAD_ANYDEPTH])

        #plt.show()


if __name__ == '__main__':
    fix_seeds()

    cfg_model, cfg_synthia, cfg_cityscapes = get_configuration()
    model = create_model(cfg_model)

    img_h1, img_w1 = cfg_cityscapes.dataset.feed_img_size[1], cfg_cityscapes.dataset.feed_img_size[0]

    cityscapes_loader, synthia_loader = create_dataloaders(cfg_cityscapes, cfg_synthia)

    path_to_save = r'C:\Users\benba\Documents\University\Masterarbeit\data\Synthia_aug_Cityscapes'
    generate_synthia_augmented_cityscapes(cityscapes_loader, synthia_loader, model, img_h1, img_w1, path_to_save)

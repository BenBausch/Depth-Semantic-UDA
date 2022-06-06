# Project Imports
from dataloaders.cityscapes.dataset_cityscapes_sequence import CityscapesSequenceDataset
from dataloaders.synthia.dataset_synthia import SynthiaRandCityscapesDataset
from cfg.config_dataset import get_cfg_dataset_defaults
from utils.constans import IGNORE_INDEX_SEMANTIC
from misc import transforms as tf_prep

# Packages
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop, GaussianBlur
import matplotlib.pyplot as plt
import random
import os
from torchvision import transforms
import copy

from PIL import Image


class SynthiaCityscapesClassMixDataset(data.Dataset):
    """
        RGB Images annotated with Depth and Semantic classes. This dataset does not provide any image sequences.
    """

    def __init__(self, mode, split, cfg):
        """
        :param mode: mode of usage, synthia only supports 'train'
        :param split: split name of the data, currently only None is supported
        :param cfg: configuration
        """
        # The dataset does not contain sequences therefore the offsets refer only to the selected RGB image(offset == 0)
        assert len(cfg.dataset.rgb_frame_offsets) == 1
        assert cfg.dataset.rgb_frame_offsets[0] == 0
        assert split is None, 'Synthia_Cityscapes_ClassMix currently does not support the spilt of the data!'
        assert mode == 'train', 'Synthia_Cityscapes_ClassMix does not support any other mode than train!'
        assert cfg.dataset.num_classes == 16, "Synthia_Cityscapes_ClassMix_Dataset only supports 16 classes!"

        super(SynthiaCityscapesClassMixDataset, self).__init__()
        self.cfg = cfg
        self.rgb_frame_offsets = cfg.dataset.rgb_frame_offsets
        self.feed_img_size = cfg.dataset.feed_img_size
        self.img_height = self.feed_img_size[1]
        self.img_width = self.feed_img_size[0]

        # load the datasets
        self.sub_datasets_config_paths = copy.deepcopy(self.cfg.dataset.sub_dataset_paths)

        self.synthia_cfg = get_cfg_dataset_defaults()
        self.synthia_cfg.merge_from_file(os.path.join(os.getcwd(),
                                                      'dataloaders',
                                                      'synthia_aug_cityscapes',
                                                      'synthia_classmix.yaml'))
        self.synthia_cfg.dataset.path = self.sub_datasets_config_paths[0]
        self.synthia_cfg.freeze()
        self.synthia_ds = SynthiaRandCityscapesDataset('train', None, self.synthia_cfg)
        self.synthia_length = len(self.synthia_ds)

        self.cityscapes_cfg = get_cfg_dataset_defaults()
        self.cityscapes_cfg.merge_from_file(os.path.join(os.getcwd(),
                                                         'dataloaders',
                                                         'synthia_aug_cityscapes',
                                                         'cityscapes_sequence_classmix.yaml'))
        self.cityscapes_cfg.dataset.path = self.sub_datasets_config_paths[1]
        self.cityscapes_cfg.freeze()
        self.cityscapes_ds = CityscapesSequenceDataset('train', None, self.cityscapes_cfg)
        self.cityscapes_length = len(self.cityscapes_ds)

        self.number_classes_to_sample = int(self.cfg.dataset.num_classes / 2)

        self.aug_params = {"brightness_jitter": cfg.dataset.augmentation.brightness_jitter,
                           "contrast_jitter": cfg.dataset.augmentation.contrast_jitter,
                           "saturation_jitter": cfg.dataset.augmentation.saturation_jitter,
                           "hue_jitter": cfg.dataset.augmentation.hue_jitter}

    def __len__(self):
        """
        Returns the number of rgb images for the mode in the split.
        :return:
        """
        return len(self.synthia_ds)  # return length of the smaller dataset

    def __getitem__(self, index):
        """
            Collects the rgb images of the sequence and the label of the index image and returns them as a dict.
            Images can be accessed by dict_variable[("rgb", offset)] with e.g. offset = 0 for the index image.
            Labels can be accessed by dict_variable["gt"].
            :param index: Index of an image with offset 0 in the sequence.
            :return: dict of images and label
        """
        #path = r'C:\Users\benba\Documents\University\Masterarbeit\Depth-Semantic-UDA\plots\dacs'

        data = {}

        # Get all required training elements
        data_synthia = self.synthia_ds.__getitem__(random.randint(0, self.synthia_length - 1))
        data_cityscapes = self.cityscapes_ds.__getitem__(random.randint(0, self.cityscapes_length - 1))

        synthia_rgb, synthia_semantic = data_synthia[("rgb", 0)], data_synthia["semantic"]
        cityscapes_rgb = data_cityscapes[("rgb", 0)]
        #data[("synthia_plot")] = synthia_rgb + torch.tensor(
        #    [[[0.314747602, 0.277402550, 0.248091921]]]).transpose(0, 2)
        #plot = Image.fromarray((data[("synthia_plot")].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        #plot.save(os.path.join(path, "synthia_plot.png"))

        #data[("cityscapes_plot")] = cityscapes_rgb + torch.tensor(
        #    [[[0.28689554, 0.32513303, 0.28389177]]]).transpose(0, 2)
        #plot = Image.fromarray((data[("cityscapes_plot")].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        #plot.save(os.path.join(path, "cityscapes_plot.png"))

        # get the two crop postions
        synthia_crop_indices = transforms.RandomCrop.get_params(
            synthia_rgb, output_size=(self.img_height, self.img_width))
        i, j, h, w = synthia_crop_indices
        crop_cityscapes = RandomCrop((self.img_height, self.img_width))

        # crop both images and sample half the semantic classes from synthia
        #data["synthia_semantic"] = synthia_semantic
        #plot = Image.fromarray((s2rgb(data[("synthia_semantic")].unsqueeze(0), num_classes=16)).astype(np.uint8))
        #plot.save(os.path.join(path, "synthia_semantic.png"))

        sampled_semantic = self.sample_semantic_gt(synthia_semantic)
        #data["sampled_semantic"] = sampled_semantic
        #plot = Image.fromarray((s2rgb(data[("sampled_semantic")].unsqueeze(0), num_classes=16)).astype(np.uint8))
        #plot.save(os.path.join(path, "sampled_semantic.png"))

        synthia_semantic = TF.crop(sampled_semantic, i, j, h, w)
        synthia_rgb = TF.crop(synthia_rgb, i, j, h, w)
        cityscapes_rgb = crop_cityscapes(cityscapes_rgb)

        # fuse cityscapes rgb into synthia rgb
        mask_cityscapes = synthia_semantic == IGNORE_INDEX_SEMANTIC # cityscapes pixel are pasted where the synthia is
        # ignored

        #plotting
        #synthia_plot = synthia_rgb + torch.tensor([[[0.314747602, 0.277402550, 0.248091921]]]).transpose(0, 2)
        #cityscapes_plot = cityscapes_rgb + torch.tensor([[[0.28689554, 0.32513303, 0.28389177]]]).transpose(0, 2)
        #synthia_plot[:, mask_cityscapes] = cityscapes_plot[:, mask_cityscapes]
        #plot = Image.fromarray(
        #    (synthia_plot.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        #plot.save(os.path.join(path, "mixed.png"))

        synthia_rgb[:, mask_cityscapes] = cityscapes_rgb[:, mask_cityscapes]

        do_color_aug = random.random() > 0.5
        mixed_sample_augmentation = self.tf_augmented_rgb_train(do_color_aug=do_color_aug,
                                                                color_aug_params=self.aug_params)

        #data[("synthia_crop_plot")] = synthia_rgb + torch.tensor([[[0.314747602, 0.277402550, 0.248091921]]]).transpose(0, 2)
        #plot = Image.fromarray((data[("synthia_crop_plot")].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        #plot.save(os.path.join(path, "synthia_crop_plot.png"))
        #data[("cityscapes_crop_plot")] = cityscapes_rgb + torch.tensor([[[0.28689554, 0.32513303, 0.28389177]]]).transpose(0, 2)
        #plot = Image.fromarray((data[("cityscapes_crop_plot")].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        #plot.save(os.path.join(path, "cityscapes_crop_plot.png"))
        data[("unaug_rgb", 0)] = cityscapes_rgb
        data[("rgb", 0)] = mixed_sample_augmentation(synthia_rgb)
        data["semantic"] = synthia_semantic
        #plot = Image.fromarray((s2rgb(data[("semantic")].unsqueeze(0), num_classes=16)).astype(np.uint8))
        #plot.save(os.path.join(path, "semantic.png"))

        return data

    def sample_semantic_gt(self, synthia_semantic):
        """
        Selects #self.number_classes_to_sample classes from the synthia semantic gt at random.
        """
        random_class_ids = random.sample(range(0, self.cfg.dataset.num_classes), self.number_classes_to_sample)
        #random_class_ids = [4, 0, 6, 7, 10, 11, 15, 12] for testing and plotting purpose

        mask = torch.zeros_like(synthia_semantic) + IGNORE_INDEX_SEMANTIC
        for class_id in random_class_ids:
            mask[synthia_semantic == class_id] = class_id

        return mask

    def tf_augmented_rgb_train(self, do_color_aug, color_aug_params):
        """
        """
        sigma = np.random.uniform(0.15, 1.15)
        kernel_size_y = int(np.floor(np.ceil(0.1 * self.img_height) - 0.5 + np.ceil(0.1 * self.img_height) % 2))
        kernel_size_x = int(np.floor(np.ceil(0.1 * self.img_width) - 0.5 + np.ceil(0.1 * self.img_width) % 2))

        return transforms.Compose(
            [
                tf_prep.ColorAug(do_aug=do_color_aug, aug_params=color_aug_params),
                GaussianBlur(kernel_size=(kernel_size_y, kernel_size_x), sigma=sigma)
            ]
        )


if __name__ == "__main__":
    torch.set_printoptions(precision=9)
    from cfg.config_dataset import get_cfg_dataset_defaults
    from torch.utils.data import DataLoader
    from utils.plotting_like_cityscapes_utils import semantic_id_tensor_to_rgb_numpy_array as s2rgb

    cfg = get_cfg_dataset_defaults()
    cfg.merge_from_file(
        r'C:\Users\benba\Documents\University\Masterarbeit\Depth-Semantic-UDA\cfg\yaml_files\train\guda_dacs\synthia_aug_cityscapes.yaml')
    cfg.freeze()

    dataset = SynthiaCityscapesClassMixDataset("train", None, cfg)
    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False)

    for idx, data in enumerate(loader):
        cityscapes_rgb = data[("unaug_rgb", 0)]
        synthia_rgb = data[("rgb", 0)]
        synthia_semantic = data["semantic"]
        fig, axs = plt.subplots(1, 3)
        synthia_plot = synthia_rgb[0] + torch.tensor([[[0.314747602, 0.277402550, 0.248091921]]]).transpose(0, 2)
        axs[0].imshow(synthia_plot.permute(1, 2, 0).numpy())
        axs[1].imshow(s2rgb(synthia_semantic, num_classes=16))
        cityscapes_plot = cityscapes_rgb[0] + torch.tensor([[[0.28689554, 0.32513303, 0.28389177]]]).transpose(0, 2)
        axs[2].imshow(cityscapes_plot.permute(1, 2, 0).numpy())
        plt.show()

    print(dataset)


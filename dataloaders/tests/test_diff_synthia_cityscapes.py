# Python Packages
import os
import unittest
import random

# Own Packages
from dataloaders.synthia.dataset_synthia import SynthiaRandCityscapesDataset
from dataloaders.cityscapes.dataset_cityscapes_semantic import CityscapesSemanticDataset
from cfg.config_training import get_cfg_dataset_defaults

# Dependencies
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import numpy as np


class TestSynthiaDataset(unittest.TestCase):
    """
    This is not really a test class, but it is simply used to visualize and plot differences in the data
    across cityscapes and synthia, to verify that they are loaded similarly
    """

    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        random.seed(0)
        path_16 = os.path.join('dataloaders', 'tests', 'data', 'synthia_16.yaml')
        cfg_16 = get_cfg_dataset_defaults()
        cfg_16.merge_from_file(path_16)
        cfg_16.freeze()
        ds_16 = SynthiaRandCityscapesDataset(mode='train', split=None, cfg=cfg_16)
        self.s_loader = DataLoader(ds_16, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
        path_16 = os.path.join('dataloaders', 'tests', 'data', 'cityscapes_semantic_16.yaml')
        cfg_16 = get_cfg_dataset_defaults()
        cfg_16.merge_from_file(path_16)
        cfg_16.freeze()
        ds_16 = CityscapesSemanticDataset('train', None, cfg=cfg_16)
        self.c_loader = DataLoader(ds_16, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    def testPlotImages(self):
        self.setUp()
        for batch_idx, data in enumerate(zip(self.s_loader, self.c_loader)):
            print(np.max(data[0][('rgb', 0)][0].numpy().transpose(1, 2, 0)))
            print(np.min(data[0][('rgb', 0)][0].numpy().transpose(1, 2, 0)))
            print(np.max(data[1][('rgb', 0)][0].numpy().transpose(1, 2, 0)))
            print(np.min(data[1][('rgb', 0)][0].numpy().transpose(1, 2, 0)))
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(data[0][('rgb', 0)][0].numpy().transpose(1, 2, 0))
            ax2.imshow(data[1][('rgb', 0)][0].numpy().transpose(1, 2, 0))
            plt.show()
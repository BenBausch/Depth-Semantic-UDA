# Python Packages
import os
import unittest
import random

# Own Packages
from dataloaders.cityscapes.dataset_cityscapes_semantic import CityscapesSemanticDataset
from cfg.config_training import get_cfg_dataset_defaults

# Dependencies
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import numpy as np


class TestCityscapesSequenceDataset(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        random.seed(0)
        path_19 = os.path.join('dataloaders', 'tests', 'data', 'cityscapes_semantic_19.yaml')
        cfg_19 = get_cfg_dataset_defaults()
        cfg_19.merge_from_file(path_19)
        cfg_19.freeze()
        ds_19 = CityscapesSemanticDataset('val', None, cfg=cfg_19)
        self.loader_19 = DataLoader(ds_19, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
        path_16 = os.path.join('dataloaders', 'tests', 'data', 'cityscapes_semantic_16.yaml')
        cfg_16 = get_cfg_dataset_defaults()
        cfg_16.merge_from_file(path_16)
        cfg_16.freeze()
        ds_16 = CityscapesSemanticDataset('val', None, cfg=cfg_16)
        self.loader_16 = DataLoader(ds_16, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    def testCityscapesIds(self):
        """
        Test if cityscapes labeling for each class is correct
        """
        for idx, data in enumerate(self.loader_19):
            # idx = 0 should be first image in validation frankfurt folder
            # class with id 9, 14, 16 no instance in synthia
            if idx == 0:
                self.assertEqual(data['semantic'][0, 577, 596].item(), 0)
                self.assertEqual(data['semantic'][0, 708, 1701].item(), 1)
                self.assertEqual(data['semantic'][0, 204, 177].item(), 2)
                self.assertEqual(data['semantic'][0, 404, 959].item(), 4)
                self.assertEqual(data['semantic'][0, 625, 45].item(), 5)
                self.assertEqual(data['semantic'][0, 225, 966].item(), 8)
                self.assertEqual(data['semantic'][0, 56, 1065].item(), 10)
                self.assertEqual(data['semantic'][0, 422, 955].item(), 11)
                self.assertEqual(data['semantic'][0, 436, 1502].item(), 13)
                self.assertEqual(data['semantic'][0, 64, 1651].item(), 250)
            if idx == 1:
                self.assertEqual(data['semantic'][0, 212, 1597].item(), 7)
                self.assertEqual(data['semantic'][0, 394, 1036].item(), 12)
                self.assertEqual(data['semantic'][0, 418, 1036].item(), 18)
            if idx == 3:
                self.assertEqual(data['semantic'][0, 276, 327 ].item(), 6)
            if idx == 7:
                self.assertEqual(data['semantic'][0, 357, 784].item(), 14)
            if idx == 8:
                self.assertEqual(data['semantic'][0, 369, 871].item(), 16)
            if idx == 9:
                self.assertEqual(data['semantic'][0, 394, 1939].item(), 9)
                self.assertEqual(data['semantic'][0, 369, 1370].item(), 15)
            if idx == 12:
                self.assertEqual(data['semantic'][0, 469, 210].item(), 17)
            if idx == 16:
                self.assertEqual(data['semantic'][0, 406, 566].item(), 3)
            break

        for idx, data in enumerate(self.loader_16):
            # idx = 0 should be first image in validation frankfurt folder
            # class with id 9, 14, 16 no instance in synthia
            if idx == 0:
                self.assertEqual(data['semantic'][0, 577, 596].item(), 0)
                self.assertEqual(data['semantic'][0, 708, 1701].item(), 1)
                self.assertEqual(data['semantic'][0, 204, 177].item(), 2)
                self.assertEqual(data['semantic'][0, 404, 959].item(), 4)
                self.assertEqual(data['semantic'][0, 625, 45].item(), 5)
                self.assertEqual(data['semantic'][0, 225, 966].item(), 8)
                self.assertEqual(data['semantic'][0, 56, 1065].item(), 9)
                self.assertEqual(data['semantic'][0, 422, 955].item(), 10)
                self.assertEqual(data['semantic'][0, 436, 1502].item(), 12)
                self.assertEqual(data['semantic'][0, 64, 1651].item(), 250)
            if idx == 1:
                self.assertEqual(data['semantic'][0, 212, 1597].item(), 7)
                self.assertEqual(data['semantic'][0, 394, 1036].item(), 11)
                self.assertEqual(data['semantic'][0, 418, 1036].item(), 15)
            if idx == 3:
                self.assertEqual(data['semantic'][0, 276, 327 ].item(), 6)
            if idx == 7:
                self.assertEqual(data['semantic'][0, 357, 784].item(), 250)  # different to loader 19
            if idx == 8:
                self.assertEqual(data['semantic'][0, 369, 871].item(), 250)  # different to loader 19
            if idx == 9:
                self.assertEqual(data['semantic'][0, 394, 1939].item(), 250)  # different to loader 19
                self.assertEqual(data['semantic'][0, 369, 1370].item(), 13)
            if idx == 12:
                self.assertEqual(data['semantic'][0, 469, 210].item(), 14)
            if idx == 16:
                self.assertEqual(data['semantic'][0, 406, 566].item(), 3)
            break


if "__main__" == __name__:
    unittest.main(warnings='ignore')

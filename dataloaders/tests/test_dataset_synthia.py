# Python Packages
import os
import unittest
import random

# Own Packages
from dataloaders.synthia.dataset_synthia import SynthiaRandCityscapesDataset
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
        path = os.path.join('dataloaders','tests','data','synthia.yaml')
        cfg = get_cfg_dataset_defaults()
        cfg.merge_from_file(path)
        cfg.freeze()
        ds = SynthiaRandCityscapesDataset(mode='train', split=None, cfg=cfg)
        self.loader_1 = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
        self.loader_3 = DataLoader(ds, batch_size=3, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    def testCityscapesIds(self):
        """
        Test if cityscapes labeling for each class is correct
        """
        for idx, data in enumerate(self.loader_1):
            # idx = 0 should be image with name 0000000.png
            # class with id 9, 14, 16 no instance in synthia
            print(idx)
            if idx == 0:
                self.assertEqual(data['semantic'][0, 577, 596].item(), 0)
                self.assertEqual(data['semantic'][0, 629, 268].item(), 1)
                self.assertEqual(data['semantic'][0, 92, 1174].item(), 2)
                self.assertEqual(data['semantic'][0, 395, 428].item(), 3)
                self.assertEqual(data['semantic'][0, 429, 515].item(), 4)
                self.assertEqual(data['semantic'][0, 391, 761].item(), 5)
                self.assertEqual(data['semantic'][0, 385, 162].item(), 8)
                self.assertEqual(data['semantic'][0, 64, 518].item(), 10)
                self.assertEqual(data['semantic'][0, 559, 92].item(), 11)
                self.assertEqual(data['semantic'][0, 492, 765].item(), 12)
                self.assertEqual(data['semantic'][0, 466, 1232].item(), 13)
                self.assertEqual(data['semantic'][0, 471, 624].item(), 17)
                self.assertEqual(data['semantic'][0, 617, 778].item(), 18)
                self.assertEqual(data['semantic'][0, 439, 171].item(), 250)
            if idx == 50:
                self.assertEqual(data['semantic'][0, 9, 330].item(), 6)
                self.assertEqual(data['semantic'][0, 249, 754].item(), 7)
                self.assertEqual(data['semantic'][0, 366, 36].item(), 15)
                break



if "__main__" == __name__:
    unittest.main(warnings='ignore')
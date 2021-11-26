import os
import glob
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from tensorboardX import SummaryWriter
import argparse
from datetime import datetime

import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import io_utils

# Reference point for all configurable options
from yacs.config import CfgNode as CN

# /----- Create a cfg node
cfg = CN()
cfg.num_epochs = 10
cfg.dataset_name = 'dummy'
cfg.model_name = 'linearNN'
cfg.freeze()

# Define a dummy neural network for testing io_utils
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.lin1 = nn.Linear(10, 10)
        self.lin2 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num

class Test_IOUtils(unittest.TestCase):
    def setUp(self):
        # Set up a Neural Network
        self.models = {}
        self.models[cfg.model_name] = NeuralNetwork()
        self.models[cfg.model_name] = self.models[cfg.model_name].cuda()
        self.optimizer = optim.SGD(self.models[cfg.model_name].parameters(), lr=0.1)

        self.path_base = os.path.join(os.path.dirname(__file__), "tmp")
        self.io_handler = io_utils.IOHandler(self.path_base)

    def test_save_load(self):
        # Delete the folder where the models are to be stored as this is a unit test and we do not want to store too
        # many of them to save memory
        if os.path.exists(self.path_base):
            shutil.rmtree(self.path_base)

        self.epoch = 0
        self.step_train = 0

        # Get the starting time of the training
        current_time = datetime.now()
        self.training_start_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")

        # Save the config file
        cfg_info = {}
        cfg_info["time"] = self.training_start_time
        cfg_info["dataset"] = cfg.dataset_name
        self.io_handler.save_cfg(cfg_info, cfg)

        # Train network and save per epoch with given frequency
        print("Start training...")
        for self.epoch in range(cfg.num_epochs):
            print("Current epoch: ", self.epoch)
            x = [1, 0, 0, 0, 1, 0, 0, 0, 1, 1]
            input = Variable(torch.Tensor([x for _ in range(10)]))
            input = input.cuda()
            out = self.models[cfg.model_name](input)

            x = [0, 1, 1, 1, 0, 1, 1, 1, 0, 0]
            target = Variable(torch.Tensor([x for _ in range(10)]))
            target = target.cuda()
            criterion = nn.MSELoss()
            loss = criterion(out, target)

            print("Loss:", loss)

            self.models[cfg.model_name].zero_grad()
            loss.backward()
            self.optimizer.step()


            if (self.epoch + 1) % 2 == 0:
                # Get information on the training
                info = {}
                info["epoch"] = self.epoch
                info["time"] = self.training_start_time
                info["dataset"] = cfg.dataset_name

                # Create checkpoint
                checkpoint = io_utils.IOHandler.gen_checkpoint(
                    self.models,
                    **{"optimizer": self.optimizer.state_dict(),
                       "opts": cfg})

                self.io_handler.save_checkpoint(info, checkpoint)

                # Now load and initialize a new model and check whether the state dictionaries are the same
                # Get the latest file that was saved first
                all_subdirs = [os.path.join(self.path_base, d) for d in os.listdir(self.path_base)
                               if os.path.isdir(os.path.join(self.path_base, d))]
                recent_subdir = max(all_subdirs, key=os.path.getmtime)

                # Create new model and optimizer
                models = dict()
                models[cfg.model_name] = NeuralNetwork()
                models[cfg.model_name] = models[cfg.model_name].cuda()
                optimizer = optim.SGD(models[cfg.model_name].parameters(), lr=0.1)

                # Load state dictionary
                checkpoint = io_utils.IOHandler.load_checkpoint(os.path.join(self.path_base, recent_subdir), 'checkpoint_epoch_{}.pth'.format(info["epoch"]))
                io_utils.IOHandler.load_weights(checkpoint, models, optimizer)

                # Compare
                for key_item_1, key_item_2 in zip(models[cfg.model_name].state_dict().items(), self.models[cfg.model_name].state_dict().items()):
                    assert torch.equal(key_item_1[1], key_item_2[1]), "Mismatch found concerning model weights!"
                    assert key_item_1[0] == key_item_2[0], "Mismatch found concerning model weight keys!"

                assert optimizer.state_dict().items() == self.optimizer.state_dict().items()


if "__main__" == __name__:
    unittest.main()

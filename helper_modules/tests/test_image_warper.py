#!/usr/bin/python
# -*- coding: latin-1 -*-
import unittest
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as Rot
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(1, '/home/petek/workspace/py_workspace/depth_estimation/depth_estimation/camera_models')
sys.path.insert(1, '/home/petek/workspace/py_workspace/depth_estimation/depth_estimation/train')

from helper_modules.image_warper import ImageWarper
from camera_models.camera_model_pinhole import PinholeCameraModel


class Test_ImageWarper(unittest.TestCase):
    def setUp(self):
        self.nof_data = 1
        self.batch_size = 1
        posObj = np.array([+0.2, -0.5, 0.0])
        posDiff = torch.from_numpy(posObj)
        rotObj = Rot.from_euler('zxy', [[0.05, 0.1, 0.3]], degrees=False).inv() # These are euler angles in the order of 1. roll, 2. pitch, 3. yaw about the fixed reference frame...
        rotDiffMat = torch.from_numpy((
            rotObj.as_matrix()))  # Achtung: rotDiffMat ist fr jedes Paar von zwei aufeinanderfolgenden Bildern anders -> entsprechend verarbeiten

        # Create 4x4 homogenoeus matrix containing both rotational and translational transformation
        self.T = torch.zeros([1, 4, 4], dtype=torch.float32)
        self.T[:, 0:3, 0:3] = rotDiffMat
        self.T[:, 0:3, 3] = posDiff
        self.T[:, 3, 0:3] = 0
        self.T[:, 3, 3] = 1

        self.path_src_img = "./data/source.png"
        self.path_tgt_img = "./data/warped.png"
        self.dataset = TestDataset(self.path_src_img, self.path_tgt_img, self.nof_data)

        # These are the attributes that are used in each camera object for testing purposes. DO NOT CHANGE THIS!
        self.img_width = 640
        self.img_height = 480
        self.fx = 554.254691
        self.fy = 554.254691
        self.cx = 320
        self.cy = 240

        # These parameters are those that are used in the corresponding Gazebo simulation
        self.camera_model = PinholeCameraModel(self.img_width, self.img_height, self.fx, self.fy, self.cx, self.cy)

        # Set up the nn-Modules
        self.image_warper = ImageWarper(self.camera_model, "cpu")

        self.to_pil = transforms.ToPILImage()

    def test_image_warper(self):
        loader = DataLoader(self.dataset, batch_size=self.batch_size)

        for i, batch_data in enumerate(loader):
            print(i, "-th run")
            src = batch_data["src"].double()
            print(torch.unique(src))
            tgt = batch_data["tgt"]
            depth = batch_data["depth"]
            src_label = batch_data["src_label"]

            tgtPlot = tgt.squeeze()
            plt.title("TARGET")
            plt.imshow(tgtPlot.permute(1, 2, 0))
            plt.show()

            # src_plt = plt.imshow(self.to_pil(src.squeeze()))
            # plt.show()
            # tgt_plt = plt.imshow(self.to_pil(tgt.squeeze()))
            # plt.show()
            # depth_plt = plt.imshow(self.to_pil(depth.squeeze()))
            # plt.show()

            warpedImg, warped_semantic = self.image_warper(src, depth, self.T, src_label)
            print(warped_semantic)
            warped_img = warpedImg.squeeze()
            plt.title("Warped image")
            plt.imshow(warped_img.numpy().transpose(1, 2, 0))
            plt.show()
            # to plot change values
            warped_semantic[0, 1, :, :][warped_semantic[0, 1, :, :] == 1] = 100
            plt.imshow(warped_semantic[0, 1, :, :])
            plt.show()


class TestDataset(Dataset):
    def __init__(self, path_src_img, path_tgt_img, nof_data):
        self.nof_data = nof_data
        self.image_paths = [
            path_src_img,
            path_tgt_img
        ]

        path_to_src_label = path_src_img[:-4] + '_label.png'
        path_to_tgt_label = path_tgt_img[:-4] + '_label.png'

        self.label_paths = [
            path_to_src_label,
            path_to_tgt_label
        ]

        self.to_tensor = transforms.ToTensor()
        self.pil2t = transforms.PILToTensor()

    def __getitem__(self, index):
        path_src = self.image_paths[index]
        path_tgt = self.image_paths[index + 1]
        path_src_label = self.label_paths[index]
        path_tgt_label = self.label_paths[index + 1]

        with open(path_src, 'rb') as f:
            img_src = Image.open(f).convert('RGB')

        with open(path_src_label, 'rb') as f:
            label_src = Image.open(f).convert('RGB')
            label_src = self.pil2t(label_src)[0, :, :].bool().long()
            #plt.imshow(label_src.permute(1, 2, 0))
            #plt.show()
            label_src = F.one_hot(label_src, 2).double().permute(2, 0, 1)

        with open(path_tgt, 'rb') as f:
            img_tgt = Image.open(f).convert('RGB')

        with open(path_tgt_label, 'rb') as f:
            label_tgt = Image.open(f).convert('RGB')
            label_tgt = self.pil2t(label_tgt)[0, :, :].bool().long()
            #plt.imshow(label_tgt.permute(1, 2, 0))
            #plt.show()
            label_tgt = F.one_hot(label_tgt, 2).double().permute(2, 0, 1)
            print(label_tgt.shape)

        test = self.to_tensor(img_tgt)

        depth_map = np.full((480, 640), 2.5)
        depth_map = np.expand_dims(depth_map, axis=0)
        depth = torch.from_numpy(depth_map)

        items = {"src": self.to_tensor(img_src),
                 "tgt": self.to_tensor(img_tgt),
                 "depth": depth,
                 "src_label": label_src,
                 "tgt_label": label_tgt}

        return items

    def __len__(self):
        return self.nof_data


if "__main__" == __name__:
    unittest.main()

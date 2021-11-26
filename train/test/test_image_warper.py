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

import torchgeometry as tgm

sys.path.insert(1, '/home/petek/workspace/py_workspace/depth_estimation/depth_estimation/camera_models')
sys.path.insert(1, '/home/petek/workspace/py_workspace/depth_estimation/depth_estimation/train')

import image_warper as warper
import camera_model_pinhole as cam

class Test_ImageWarper(unittest.TestCase):
    def setUp(self):
        self.nof_data = 1
        self.batch_size = 1
        posObj = (-1)*np.array([0.5, 0.5, -5]) # Pose difference from pose 1 to 2 inside of the array only
        #posObj = (-1)*np.array([+0.2, -0.5, 0.0])
        posDiff = torch.from_numpy(posObj)
        rotObj = Rot.from_euler('xyz', [[0, 0, 0]], degrees=False).inv()
        #rotObj =  Rot.from_euler('zxy', [[0.05, -0.1, -0.3]], degrees=False).inv()#.inv() # These are euler angles in the order of 1. roll, 2. pitch, 3. yaw about the fixed reference frame...
        #rotObj =  Rot.from_euler('zxy', [[0.05, -0.1, -0.3]], degrees=False)#.inv() # These are euler angles in the order of 1. roll, 2. pitch, 3. yaw about the fixed reference frame...
        #rotObj =  Rot.from_euler('xyz', [[0.05, 0.1, 0.3]], degrees=False).inv() # These are euler angles in the order of 1. roll, 2. pitch, 3. yaw about the fixed reference frame...
        rotDiffMat = torch.from_numpy((rotObj.as_matrix())) # Achtung: rotDiffMat ist für jedes Paar von zwei aufeinanderfolgenden Bildern anders -> entsprechend verarbeiten

        # Create 4x4 homogenoeus matrix containing both rotational and translational transformation
        self.T = torch.zeros([1, 4, 4], dtype=torch.float32)
        self.T[:, 0:3, 0:3] = rotDiffMat
        self.T[:, 0:3, 3] = posDiff
        self.T[:, 3, 0:3] = 0
        self.T[:, 3, 3] = 1

        self.path_src_img = "./data/source.png"
        self.path_tgt_img = "./data/warped.png"
        #self.depth = 2.5 # The plane has a unique depth of 2.5 w.r.t the camera
        self.dataset = TestDataset(self.path_src_img, self.path_tgt_img, self.nof_data)

        # These are the attributes that are used in each camera object for testing purposes. DO NOT CHANGE THIS!
        self.img_width = 640
        self.img_height = 480
        self.fx = 554.254691
        self.fy = 554.254691
        self.cx = 320
        self.cy = 240

        # These parameters are those that are used in the corresponding Gazebo simulation
        self.camera_model = cam.PinholeCameraModel(self.img_width, self.img_height, self.fx, self.fy, self.cx, self.cy)

        # Set up the nn-Modules
        self.image_warper = warper.ImageWarper(self.camera_model, "cpu")

        self.to_pil = transforms.ToPILImage()

    def test_image_warper(self):
        loader = DataLoader(self.dataset, batch_size=self.batch_size)

        for i, batch_data in enumerate(loader):
            print(i, "-th run")
            src = batch_data["src"].double()
            tgt = batch_data["tgt"]
            depth = batch_data["depth"]

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

            warpedImg = self.image_warper(src, depth, self.T)
            warped_img = warpedImg.squeeze()
            plt.title("Warped image")
            plt.imshow(warped_img)
            plt.show()


class TestDataset(Dataset):
    def __init__(self, path_src_img, path_tgt_img, nof_data):
        self.nof_data = nof_data
        self.image_paths = [
            path_src_img,
            path_tgt_img
        ]

        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        path_src = self.image_paths[index]
        path_tgt = self.image_paths[index+1]

        with open(path_src, 'rb') as f:
            img_src = Image.open(f).convert('RGB')

        with open(path_tgt, 'rb') as f:
            img_tgt = Image.open(f).convert('RGB')

        test = self.to_tensor(img_tgt)

        depth_map = np.full((480, 640), 2.5)
        depth_map = np.expand_dims(depth_map, axis=0)
        depth = torch.from_numpy(depth_map)
        # depth = self.to_tensor(Image.fromarray(np.uint8(depth_map)).convert('L'))
        items = {"src": self.to_tensor(img_src), "tgt": self.to_tensor(img_tgt), "depth": depth}

        return items

    def __len__(self):
        return self.nof_data


if "__main__" == __name__:
        unittest.main()
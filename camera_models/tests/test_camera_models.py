#!/usr/bin/python
# -*- coding: latin-1 -*-
import os
import torch
import unittest
import numpy as np
from PIL import Image, ImageDraw, ImageChops

import camera_models.camera_model_pinhole as cam

class Test_PinholeCameraModel(unittest.TestCase):
    def setUp(self):
        # These are the attributes that are used in each camera object for testing purposes. DO NOT CHANGE THIS!
        self.img_width = 640
        self.img_height = 480
        self.fx = 554.254691
        self.fy = 554.254691
        self.cx = 320
        self.cy = 240

        # Get RGB GT image created in Gazebo for testing the reprojection
        path_src_img = "data/source.png"
        self.path_pred_img = "data/predicted.png"

        assert os.path.exists(path_src_img), "The file {} does not exist!".format(path_file)
        with open(path_src_img, 'rb') as f:
            self.imgSrc = Image.open(f).convert('RGB')

        self.xRedPlaneLeft = torch.tensor(-0.5)
        self.xRedPlaneRight = torch.tensor(0.5)
        self.yRedPlaneUpper = torch.tensor(-0.5)
        self.yRedPlaneLower = torch.tensor(0.5)
        self.boxColor = (102, 0, 0)
        self.backgroundRgb = (178, 178, 178)

        # These parameters are those that are used in the corresponding Gazebo simulation
        self.PinholeFromParams = cam.PinholeCameraModel(self.img_width, self.img_height, self.fx, self.fy, self.cx, self.cy)
        self.PinholeFromFile = None
        self.PinholeFromRosMsg = None

    def test_fx(self):
        assert self.PinholeFromParams.fx() == self.fx

    def test_fy(self):
        assert self.PinholeFromParams.fy() == self.fy

    def test_cx(self):
        assert self.PinholeFromParams.cx() == self.cx

    def test_cy(self):
        assert self.PinholeFromParams.cy() == self.cy

    def test_image_size(self):
        assert self.PinholeFromParams.image_size() == (self.img_width, self.img_height)

    def test_model_type(self):
        assert isinstance(self.PinholeFromParams, cam.PinholeCameraModel)

    def test_intrinsics_dict(self):
        assert self.PinholeFromParams.intrinsics_dict() == {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}

    def test_get_image_point(self):
        # Sample the whole 3d plane as placed in the Gazebo simulation
        step_size = 0.001

        # Due to small inaccuracies in the simulation and imperfections when transforming from 3d to 2d (float to int)
        # a tolerance of eps is added to avoid assertions caused by imperfections
        eps = 0.005

        # That works here, as the box is symmetric and the number of elements in pts_3d_x and pts_3d_y is the same
        # We kept it like this for the sake of simplicity
        pts_3d_x = torch.arange(self.xRedPlaneLeft, self.xRedPlaneRight, step_size)
        pts_3d_y = torch.arange(self.yRedPlaneUpper, self.yRedPlaneLower, step_size)
        depth = 2.5

        # Create grid
        xx, yy = torch.meshgrid(pts_3d_x, pts_3d_y)

        # Reproject the 3d points into the image (there might be reprojected pixels that lie outside of the image)
        u2d, v2d = self.PinholeFromParams.get_image_point(xx, yy, depth)

        # Create new image for predictions
        imgPredicted = Image.new("RGB", (self.img_width, self.img_height), self.backgroundRgb)
        draw = ImageDraw.Draw(imgPredicted)

        u2d = u2d.flatten()
        v2d = v2d.flatten()

        draw.point(list(torch.from_numpy(np.array((np.array(u2d), np.array(v2d))).T.flatten())), self.boxColor)
        imgPredicted.save(self.path_pred_img)

        pred_np = np.array(imgPredicted)
        src_np = np.array(self.imgSrc)
        diff = np.fabs(np.subtract(src_np[:], pred_np[:]))

        nof_nonzeros = np.count_nonzero(diff)

        # This is not a perfect test, as we just require 99% of the pixels to be inside. There might be differences
        # between the predicted and the source image on the box boundaries and due to the sampling which might yield
        # some unprojected points inside of the red box...
        assert(nof_nonzeros/(self.img_width*self.img_height) < 0.01)

    def test_get_viewing_ray(self):
        # Unique depth of 2.5 is used in the gazebo simulation
        depth = 2.5

        # Get edges of the red plane by reprojection from 3d to 2d
        u2d_min, v2d_min = self.PinholeFromParams.get_image_point(self.xRedPlaneLeft, self.yRedPlaneUpper, depth)
        u2d_max, _ = self.PinholeFromParams.get_image_point(self.xRedPlaneRight, self.yRedPlaneLower, depth)
        _, v2d_max = self.PinholeFromParams.get_image_point(self.xRedPlaneLeft, self.yRedPlaneLower, depth)

        # Get back the 3d coordinates from the 2d edge points
        ray_ll = torch.tensor(self.PinholeFromParams.get_viewing_ray(u2d_min, v2d_max))
        ray_lr = torch.tensor(self.PinholeFromParams.get_viewing_ray(u2d_max, v2d_max))
        ray_ul = torch.tensor(self.PinholeFromParams.get_viewing_ray(u2d_min, v2d_min))
        ray_ur = torch.tensor(self.PinholeFromParams.get_viewing_ray(u2d_max, v2d_min))

        # Check if the rays have a unit norm
        np.testing.assert_allclose(np.linalg.norm(ray_ll), 1.0, rtol=1e-5, atol = 0)
        np.testing.assert_allclose(np.linalg.norm(ray_lr), 1.0, rtol=1e-5, atol = 0)
        np.testing.assert_allclose(np.linalg.norm(ray_ul), 1.0, rtol=1e-5, atol = 0)
        np.testing.assert_allclose(np.linalg.norm(ray_ur), 1.0, rtol=1e-5, atol = 0)

        # Compare the edge points of the estimated plane with GT points
        pt_ll = ray_ll*depth/ray_ll[2]
        GT_ll = [self.xRedPlaneLeft, self.yRedPlaneLower, depth]
        np.testing.assert_allclose(pt_ll, GT_ll, rtol=1e-5, atol=0)

        pt_lr = ray_lr*depth/ray_lr[2]
        GT_lr = [self.xRedPlaneRight, self.yRedPlaneLower, depth]
        np.testing.assert_allclose(pt_lr, GT_lr, rtol=1e-5, atol=0)

        pt_ul = ray_ul*depth/ray_ul[2]
        GT_ul = [self.xRedPlaneLeft, self.yRedPlaneUpper, depth]
        np.testing.assert_allclose(pt_ul, GT_ul, rtol=1e-5, atol=0)

        pt_ur = ray_ur*depth/ray_ur[2]
        GT_ur = [self.xRedPlaneRight, self.yRedPlaneUpper, depth]
        np.testing.assert_allclose(pt_ur, GT_ur, rtol=1e-5, atol=0)

if "__main__" == __name__:
    unittest.main()

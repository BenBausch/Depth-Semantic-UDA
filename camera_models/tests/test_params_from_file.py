import sys
import os
import unittest

sys.path.insert(1, '/home/petek/workspace/py_workspace/depth_estimation/depth_estimation/camera_models')
import camera_model_pinhole as cam

class Test_ReadFromFile(unittest.TestCase):
    def setUp(self):
        # These are the attributes that are used in each camera object for testing purposes. DO NOT CHANGE THIS!
        self.img_width = 640
        self.img_height = 480
        self.fx = 554.254691
        self.fy = 554.254691
        self.cx = 320
        self.cy = 240

        # Get RGB GT image created in Gazebo for testing the reprojection
        path_calib_file = "data/calib.txt"

        # Set the camera model once manually and once from file
        self.PinholeFromParams = cam.PinholeCameraModel(self.img_width, self.img_height, self.fx, self.fy, self.cx, self.cy)
        self.PinholeFromFile = cam.PinholeCameraModel.fromFile(path_calib_file)
        
    def test_readFromFile(self):
        self.assertAlmostEqual(self.PinholeFromFile.img_width, self.PinholeFromParams.img_width)
        self.assertAlmostEqual(self.PinholeFromFile.img_height, self.PinholeFromParams.img_height)
        self.assertAlmostEqual(self.PinholeFromFile.intrinsics["fx"], self.PinholeFromParams.intrinsics["fx"])
        self.assertAlmostEqual(self.PinholeFromFile.intrinsics["fy"], self.PinholeFromParams.intrinsics["fy"])
        self.assertAlmostEqual(self.PinholeFromFile.intrinsics["cx"], self.PinholeFromParams.intrinsics["cx"])
        self.assertAlmostEqual(self.PinholeFromFile.intrinsics["cy"], self.PinholeFromParams.intrinsics["cy"])

if "__main__" == __name__:
    unittest.main()
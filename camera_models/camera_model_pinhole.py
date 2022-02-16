"""Provide different functionalities based on the pinhole camera model
Attention: The pinhole camera model assumes rectified (and undistorted) images.
"""

import torch
import torch.nn as nn

import numpy as np

from camera_models import camera_model


class PinholeCameraModel(camera_model.CameraModel):
    def __init__(self, img_width, img_height, fx, fy, cx, cy):
        super(PinholeCameraModel, self).__init__()
        # if not isinstance(img_width, int) or img_width < 0:
        #     raise TypeError("You have to specify an unsigned integer value for the image width! Input type {}".format(type(img_width)))
        # if not isinstance(img_height, int) or img_height < 0:
        #     raise TypeError("You have to specify an unsigned integer value for the image height! Input type {}".format(type(img_height)))
        assert fx >= 0, "fx < 0 is not allowed"
        assert fy >= 0, "fy < 0 is not allowed"
        assert cx >= 0, "cx < 0 is not allowed"
        assert cy >= 0, "cy < 0 is not allowed"

        self.img_width = img_width
        self.img_height = img_height

        # Initialize the intrinsic parameters
        self.intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

    @classmethod
    def fromCameraInfoMsg(cls, msg):
        """
        Set the camera parameters from the corresponding ros message (useful when working with KaLiMU rosbags)
        :param msg: ros message of type 'sensor_msgs/CameraInfo.msg'
        """
        #
        npmat = np.matrix(msg.P, dtype='float32')
        npmat.resize((3, 4))
        intrinsics_matrix = torch.from_numpy(npmat)[:3, :3]

        fx = intrinsics_matrix[0, 0]
        fy = intrinsics_matrix[1, 1]
        cx = intrinsics_matrix[0, 2]
        cy = intrinsics_matrix[1, 2]

        return cls(msg.width, msg.height, fx, fy, cx, cy)

    def get_scaled_model(self, scale_u, scale_v):
        return PinholeCameraModel(self.img_width * scale_u,
                                  self.img_height * scale_v,
                                  self.fx() * scale_u,
                                  self.fy() * scale_v,
                                  self.cx() * scale_u,
                                  self.cy() * scale_v)


    @classmethod
    def fromFile(cls, path_file):
        """
        Source: https://github.com/hunse/kitti/blob/master/kitti/data.py
        Set the camera parameters from the corresponding file (useful when working with open-source datasets like
        KITTI or NYUv2)
        :return:
        """
        float_chars = set("0123456789.e+- ")

        data = {}
        with open(path_file, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                value = value.strip()
                data[key] = value
                if float_chars.issuperset(value):
                    # try to cast to float array
                    try:
                        data[key] = np.array(list(map(float, value.split(' '))))
                    except ValueError:
                        pass  # casting error: data[key] already eq. value, so pass

        # Check whether the correct camera model is specified (i.e. whether the camera model name in the calibration
        # file indeed fits the type of the instantiated object)
        assert data["type"] == "pinhole"

        return cls(data["img_width"][0].astype(np.int32), data["img_height"][0].astype(np.int32), \
                   data["fx"][0].astype(np.float64), data["fy"][0].astype(np.float64), \
                   data["cx"][0].astype(np.float64), data["cy"][0].astype(np.float64))

    def get_image_point(self, x3d, y3d, z3d):
        """Computes the 2d image coordinates of the incoming 3d world point(s)
        :param x3d, y3d, z3d: x,y,z coordinates of the incoming 3d world point(s)
        :return: u,v coordinates of the computed 2d image point(s)
        """
        u2d = (x3d/z3d) * self.intrinsics["fx"] + self.intrinsics["cx"]
        v2d = (y3d/z3d) * self.intrinsics["fy"] + self.intrinsics["cy"]
        return u2d, v2d

    def get_viewing_ray(self, u2d, v2d):
        """Computes the viewing ray(s) of the incoming image point(s)
        :param u2d, v2d: u,v coordinates of the incoming image point(s)
        :return: ray_x, ray_y, ray_z: x,y,z coordinates of the unit vector(s) representing the outgoing ray(s)
        """
        # Compute a vector that points in the direction of the viewing ray (assuming a depth of 1)
        ray_x = (u2d - self.intrinsics["cx"]) / self.intrinsics["fx"]
        ray_y = (v2d - self.intrinsics["cy"]) / self.intrinsics["fy"]
        ray_z = 1.0

        # Compute the norm of the ray vector #
        norm = torch.sqrt(ray_x ** 2 + ray_y ** 2 + ray_z ** 2)

        # Normalize the ray to obtain a unit vector
        ray_x /= norm
        ray_y /= norm
        ray_z /= norm

        return [ray_x, ray_y, ray_z]

    def image_size(self):
        """Returns the full image size the camera model is created for"""
        return self.img_width, self.img_height

    def intrinsics_dict(self):
        """Returns a dictionary containing the intrinsic parameters of the camera: fx, fy, cx, cy"""
        return self.intrinsics

    def model_type(self):
        """Returns the model type of the camera"""
        return type(self)

    def fx(self):
        return self.intrinsics["fx"]

    def fy(self):
        return self.intrinsics["fy"]

    def cx(self):
        return self.intrinsics["cx"]

    def cy(self):
        return self.intrinsics["cy"]

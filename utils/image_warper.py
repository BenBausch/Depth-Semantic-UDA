"""Provide the image warping layer based on a pre-defined camera model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class _PointcloudToImage(nn.Module):
    """Reprojects all pointclouds of the batch into the image and returns a new batch of the corresponding 2d image points
    """
    def __init__(self, camera_model):
        super(_PointcloudToImage, self).__init__()
        self.camera_model = camera_model
        self.img_width, self.img_height = camera_model.image_size()

    def forward(self, batch_pcl):
        # Get the data from batch of depth images (1 dim for batch, 2 dims for matrix and 1 for x,y,z values -> dims() = 4)
        # Attention,
        assert batch_pcl.dim() == 4, 'The input pointcloud has {} dimensions which is != 4'.format(batch_pcl.dim())
        assert batch_pcl.size(1) == 3, 'The input pointcloud has {} channels which is != 3'.format(batch_pcl.size(1))

        batch_size = batch_pcl.size(0)
        x3d = batch_pcl[:, 0, :, :].view(batch_size, -1)
        y3d = batch_pcl[:, 1, :, :].view(batch_size, -1)
        z3d = batch_pcl[:, 2, :, :].clamp(min=1e-5).view(batch_size, -1)

        # Compute the pixel coordinates
        u2d, v2d = self.camera_model.get_image_point(x3d, y3d, z3d)

        # Normalize the coordinates to [-1,+1] as required for grid_sample
        u2d_norm = (u2d / (self.img_width - 1) - 0.5) * 2
        v2d_norm = (v2d / (self.img_height - 1) - 0.5) * 2

        # Put the u2d_norm and v2d_norm vectors together and reshape them
        pixel_coordinates = torch.stack([u2d_norm, v2d_norm], dim = 2)  # dim: batch_size, H*W, 2
        pixel_coordinates = pixel_coordinates.view(batch_size, self.img_height, self.img_width, 2)

        return pixel_coordinates

class _ImageToPointcloud(nn.Module):
    """Projects all images of the batch into the 3d world (batch of pointclouds)
    """
    def __init__(self, camera_model, device="cuda"):
        super(_ImageToPointcloud, self).__init__()
        # Define a grid of pixel coordinates for the corresponding image size. Each entry defines specific grid
        # pixel coordinates for which the viewing ray is to be computed
        self.camera_model = camera_model
        img_width, img_height = camera_model.image_size()
        u2d_vals = torch.arange(start=0, end=img_width).expand(img_height, img_width).float().to(device)
        v2d_vals = torch.arange(start=0, end=img_height).expand(img_width, img_height).t().float().to(device)

        self.rays_x, self.rays_y, self.rays_z = camera_model.get_viewing_ray(u2d_vals, v2d_vals)

    def forward(self, batch_depth):
        assert batch_depth.dim() == 4, 'The input batch of depth maps has {} dimensions which is != 4'\
            .format(batch_depth.dim())
        assert batch_depth.size(1) == 1, 'The input batch of depth maps has {} channels which is != 1'\
            .format(batch_depth.size(1))
        x3d = batch_depth/abs(self.rays_z)*self.rays_x
        y3d = batch_depth/abs(self.rays_z)*self.rays_y
        z3d = batch_depth/abs(self.rays_z)*self.rays_z

        return torch.cat((x3d, y3d, z3d), dim=1)

class CoordinateWarper(nn.Module):
    def __init__(self, camera_model, device):
        super(CoordinateWarper, self).__init__()
        self.device = device
        self.image_to_pointcloud = _ImageToPointcloud(camera_model, device)
        self.pointcloud_to_image = _PointcloudToImage(camera_model)
        self.img_width, self.img_height = camera_model.image_size()

    def forward(self, batch_depth_map, T):
        """
        :param pointcloud: batch of pointclouds
        :param T: Transformation matrix
        :return:
        Attention: R and t together denote a transformation rule which transforms the pointcloud from the camera
        coordinate system of the source image to that one of the target image. So it's the inverted transformation
        from the source pose to the target pose
        """
        assert batch_depth_map.dim() == 4, 'The input batch of depth maps has {} dimensions which is != 4'.format(batch_depth_map.dim())
        assert batch_depth_map.size(1) == 1, 'The input batch of depth maps has {} channels which is != 1'.format(batch_depth_map.size(1))

        # Reproject all image pixel coordinates into the 3d world (pointcloud)
        image_as_pointcloud = self.image_to_pointcloud(batch_depth_map)

        # Transform the pointcloud to homogeneous coordinates
        ones = nn.Parameter(torch.ones(batch_depth_map.size(0), 1, self.img_height, self.img_width, device=self.device), requires_grad=False)
        image_as_pointcloud_homogeneous = torch.cat([image_as_pointcloud, ones], 1)

        # Transform the obtained pointcloud into the local coordinate system of the target camera pose (homogeneous)
        transformed_pointcloud = torch.bmm(T.double(), image_as_pointcloud_homogeneous.view(
            batch_depth_map.size(0), 4, -1).double())
        transformed_pointcloud = transformed_pointcloud.view(-1, 4, self.img_height, self.img_width)

        # Transform back to Euclidean coordinates
        transformed_pointcloud = transformed_pointcloud[:, :3, :, :]/transformed_pointcloud[:, 3, :, :].unsqueeze(1)

        # Compute pixel_coordinates, which includes associations from each pixel of the source image to the target image
        pixel_coordinates = self.pointcloud_to_image(transformed_pointcloud)

        return pixel_coordinates

class ImageWarper(nn.Module):
    def __init__(self, camera_model, device):
        super(ImageWarper, self).__init__()
        self.coordinate_warper = CoordinateWarper(camera_model, device)

    def forward(self, batch_src_img, batch_depth_map, T):
        """
        :param pointcloud: batch of pointclouds
        :param T: Transformation matrix
        :return:
        Attention: R and t together denote a transformation rule which transforms the pointcloud from the camera
        coordinate system of the source image to that one of the target image. So it's the inverted transformation
        from the source pose to the target pose
        """
        assert batch_src_img.dim() == 4, 'The input batch of source images has {} dimensions which is != 4'.format(batch_src_img.dim())
        assert batch_src_img.size(1) == 3, 'The input batch of source images has {} channels which is != 3'.format(batch_src_img.size(1))

        pixel_coordinates = self.coordinate_warper(batch_depth_map, T)

        # ToDo: Here we use as padding mode "border" to account for pixels that are out of boundary. We could actually
        #  detach them completely from the computation graph (not very clever either...) Using border is not very
        #  useful, as the outer regions being padded are quite big. Detaching them may help...
        warped_image = F.grid_sample(batch_src_img, pixel_coordinates, padding_mode="border")

        return warped_image

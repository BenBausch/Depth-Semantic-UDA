import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.image_warper import ImageWarper
from utils.image_warper import CoordinateWarper


def get_loss(loss_name, *args, **kwargs):
    """
    Source: https://github.com/wvangansbeke/Sparse-Depth-Completion/blob/master/Loss/loss.py
    """
    if loss_name not in allowed_losses():
        raise NotImplementedError('Loss functions {} is not yet implemented'.format(loss_name))
    else:
        return loss_dict[loss_name](*args, **kwargs)


def allowed_losses():
    """
    Source: https://github.com/wvangansbeke/Sparse-Depth-Completion/blob/master/Loss/loss.py
    """
    return loss_dict.keys()


# This can be used for both sparse and dense depth supervision. Sparse: Set the depth values for those pixels with
# no measurements to zero
class L1LossDepth(nn.Module):
    def __init__(self):
        super(L1LossDepth, self).__init__()

    def forward(self, prediction, target):
        assert prediction.dim() == target.dim(), "Inconsistent dimensions in L1Loss between prediction and target"
        mask = (target > 0).detach()
        abs_diff = (prediction-target)[mask].abs()
        loss = abs_diff.mean()
        return loss


class SilogLossDepth(nn.Module):
    def __init__(self, weight=0.85):
        super(SilogLossDepth, self).__init__()

        self.weight = weight

    def forward(self, pred, target):
        mask = (target > 0).detach()
        log_diff = (torch.log(pred) - torch.log(target))[mask]

        silog_var = torch.mean(log_diff ** 2)
        silog_wsme = self.weight * (log_diff.mean() ** 2)

        return silog_var - silog_wsme


class L1LossPixelwise(nn.Module):
    def __init__(self):
        super(L1LossPixelwise, self).__init__()

    def forward(self, prediction, target):
        assert prediction.dim() == target.dim(), "Inconsistent dimensions in L1Loss between prediction and target"
        loss = (prediction-target).abs()
        return loss


# This can also be used for both sparse and dense depth supervision. Sparse: Set the depth values for those pixels with
# no measurements to zero
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, prediction, target):
        assert prediction.dim() == target.dim(), "Inconsistent dimensions in MSELoss between prediction and target"
        mask = (target > 0).detach()
        abs_squared_diff = torch.square((prediction-target)[mask])
        loss = abs_squared_diff.mean()
        return loss


class EdgeAwareSmoothnessLoss(nn.Module):
    def __init__(self):
        super(EdgeAwareSmoothnessLoss, self).__init__()

    def forward(self, inv_depth_map, image):
        # Compute the gradients
        abs_grad_inv_depth_x = (inv_depth_map[:, :, :, 1:] - inv_depth_map[:, :, :, :-1]).abs()
        abs_grad_inv_depth_y = (inv_depth_map[:, :, 1:, :] - inv_depth_map[:, :, :-1, :]).abs()

        abs_grad_image_x = ((image[:, :, :, 1:] - image[:, :, :, :-1]).abs()).mean(1, keepdim=True)
        abs_grad_image_y = ((image[:, :, 1:, :] - image[:, :, :-1, :]).abs()).mean(1, keepdim=True)

        # Compute the final loss
        loss_x = abs_grad_inv_depth_x*torch.exp(-abs_grad_image_x)
        loss_y = abs_grad_inv_depth_y*torch.exp(-abs_grad_image_y)

        loss = loss_x.mean() + loss_y.mean()

        return loss


class SSIMLoss(nn.Module):
    def __init__(self, window_size=3):
        super(SSIMLoss, self).__init__()
        padding = window_size // 2

        self.mu_pool = nn.AvgPool2d(window_size, padding)
        self.sig_pool = nn.AvgPool2d(window_size, padding)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, src_img, tgt_img):
        x = self.refl(src_img)
        y = self.refl(tgt_img)

        mu_x = self.mu_pool(x)
        mu_y = self.mu_pool(y)

        sigma_x = self.sig_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_pool(x * y) - mu_x * mu_y

        ssim_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return (torch.clamp((1 - ssim_n / ssim_d) / 2, 0, 1))


# TODO: Self.sup is using a mask. What for? -> Use mask to exclude black pixels
class ReconstructionLoss(nn.Module):
    # Attention, we use a camera model here, that is normalized by the image size, in order to be able to compute
    # reprojections for each possible size of the image
    # ref_img_width and ref_img_height denote the original image sizes without scalings
    def __init__(self, normalized_camera_model, num_scales, device, ssim_kernel_size=3):
        super(ReconstructionLoss, self).__init__()
        self.num_scales = num_scales
        self.l1_loss_pixelwise = L1LossPixelwise()
        self.ssim = SSIMLoss(ssim_kernel_size)
        self.image_warpers = {}
        self.scaling_modules = {}

        # Get camera models that are adapted to the actual (scaled) image sizes
        for i in range(self.num_scales):
            s = 2 ** -i
            k_s = 2 ** i
            camera_model_ = normalized_camera_model.get_scaled_model(s, s)
            self.image_warpers[i] = ImageWarper(camera_model_, device)
            self.scaling_modules[i] = torch.nn.AvgPool2d(kernel_size=k_s, stride=k_s)

    # ToDo: Exclude "black" pixels as in selfsup and maybe use a mask for excluding pixels with depth supervision
    #  The problem with black pixels might get solved by an appropriate interpolation method!! (Comp. MD2 with selfsup)
    def forward(self, batch_data, pred_depth, poses, rgb_frame_offsets, use_automasking, use_ssim, alpha=0.85, mask=None):
        assert rgb_frame_offsets[0] == 0  # This should always be zero to make sure that the first id corresponds to the current rgb image the depth is predicted for

        total_loss = 0.0

        # Scale down the depth image (scaling must be part of the optimization graph!)
        for s in range(self.num_scales):
            reconstruction_losses_s = []
            identity_losses_s = []

            scaled_depth = self.scaling_modules[s](pred_depth)
            scaled_tgt_img = self.scaling_modules[s](batch_data[("rgb", 0)])

            for frame_id in rgb_frame_offsets[1:]:
                scaled_adjacent_img_ = self.match_sizes(batch_data[("rgb", frame_id)], scaled_depth.shape)
                warped_scaled_adjacent_img_ = self.image_warpers[s](scaled_adjacent_img_, scaled_depth, poses[frame_id])

                rec_l1_loss = self.l1_loss_pixelwise(warped_scaled_adjacent_img_, scaled_tgt_img).mean(1, True)
                rec_ssim_loss = self.ssim(warped_scaled_adjacent_img_, scaled_tgt_img).mean(1, True)

                # Compute pixelwise reconstruction loss by combining photom. loss and ssim loss
                reconstruction_losses_s.append((1-alpha)*rec_l1_loss + alpha*rec_ssim_loss if use_ssim else rec_l1_loss)

                # ToDo: Actually, we shouldn't only take the minimum but exclude those pixels where the identity loss is smaller!!
                #  CHANGE THIS! Md2 claims that this is right but I don't think so... Implement both and compare!
                if use_automasking:
                    id_l1_loss = self.l1_loss_pixelwise(scaled_adjacent_img_, scaled_tgt_img).mean(1, True)
                    id_ssim_loss = self.ssim(scaled_adjacent_img_, scaled_tgt_img).mean(1, True)
                    identity_loss = (1-alpha)*id_l1_loss + alpha*id_ssim_loss if use_ssim else id_l1_loss
                    identity_loss += torch.randn(identity_loss.shape, device=identity_loss.device) * 0.00001
                    identity_losses_s.append(identity_loss)

            # Combine the identity and reconstruction losses and get the minimum out of it for handling
            # disocclusions/occlusions (reconstruction) and sequences where the scene is static or objects are moving
            # with the same velocity
            reconstruction_losses_s = torch.cat(reconstruction_losses_s, 1)

            if use_automasking:
                identity_losses_s = torch.cat(identity_losses_s, 1)
                combined_losses_s = torch.cat((reconstruction_losses_s, identity_losses_s), dim=1)
            else:
                combined_losses_s = reconstruction_losses_s

            final_loss_per_pixel_s, _ = torch.min(combined_losses_s, dim=1)
            final_loss_s = final_loss_per_pixel_s.mean()

            # ToDo: This is used when downscaling is used, but you might use the upscaling of the depth map as in Md2
            #  later. Do not forget to change this then
            total_loss += final_loss_s / (2 ** s)

        return total_loss

    def match_sizes(self, image, target_shape, mode='bilinear', align_corners=True):
        if len(target_shape) > 2:
            target_shape = target_shape[-2:]

        return F.interpolate(image, size=target_shape, mode=mode, align_corners=align_corners)


# ToDo: Actually we need some curriculum learning or something like that, as since we use the poses here, the pose
#  network can minimize this loss easily by just forcing the pose difference to be extremely small. This way the
#  reprojection will lead to the same value! (zero pose, same pixel, depth doesn't matter anymore...)
class DepthReprojectionLoss(nn.Module):
    # Attention, we use a camera model here, that is normalized by the image size, in order to be able to compute
    # reprojections for each possible size of the image
    # ref_img_width and ref_img_height denote the original image sizes without scalings
    def __init__(self, scale_u, scale_v, normalized_camera_model, device):
        super(DepthReprojectionLoss, self).__init__()
        camera_model_ = normalized_camera_model.get_scaled_model(scale_u, scale_v)
        self.coordinate_warper = CoordinateWarper(camera_model_, device)
        self.l1_loss = L1LossDepth()

    def forward(self, pred_depth, gt_depth, T):
        # Compute the warped pixel coordinates based on the estimated depth
        # ToDo: Detaching T also helps, as the pose network doesn't adapt its weights according to this weight and hence
        #  the pose is not set to zero to minimize this loss. But there might be a slight loss of consistency then.
        #  best method would be to train with this loss once the whole network is pretrained. But training from scratch
        #  doesn't make sense
        # pose = T.detach()
        # pose[:, 0, 3] = 5
        # pose[:, 1, 3] = 5
        # pose[:, 2, 3] = 5

        coordinates_pred = self.coordinate_warper(pred_depth, T)

        # Compute the warped pixel coordinates based on the measured (GT) depth
        coordinates_gt = self.coordinate_warper(gt_depth, T)

        return self.l1_loss(coordinates_pred, coordinates_gt) # ToDo: Check this


loss_dict = {
    'reconstruction': ReconstructionLoss,
    'ssim': SSIMLoss,
    'edge_smooth': EdgeAwareSmoothnessLoss,
    'mse': MSELoss,
    'l1_depth': L1LossDepth,
    'silog_depth': SilogLossDepth,
    'depth_reprojection': DepthReprojectionLoss,
    'l1_pixelwise': L1LossPixelwise,
    'cross_entropy': nn.CrossEntropyLoss
}

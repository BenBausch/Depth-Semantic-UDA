import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from helper_modules.image_warper import ImageWarper, _ImageToPointcloud
from helper_modules.image_warper import CoordinateWarper
from utils.constans import IGNORE_INDEX_SEMANTIC

# ----------------------------------------------Semantic Losses---------------------------------------------------------
class WeightedCrossEntropy(nn.Module):
    """
    A simple wrapper for the cross entropy loss implemented in pytorch requiring weights
    """

    def __init__(self, weights, ignore_index=-100):
        self.weight = weights
        self.criterion = nn.CrossEntropyLoss(reduction='mean', weight=self.weights, ignore_index=ignore_index)

    def __call__(self, prediction, target):
        return self.criterion(prediction, target)


class BootstrappedCrossEntropy(nn.Module):
    """
        Calculates the cross entropy of the k pixels with the lowest confident prediction.
    """

    def __init__(self, img_height, img_width, r=0.3, ignore_index=-100, start_decay_epoch=None, end_decay_epoch=None):
        """
        :param img_height: height of the input image to the network
        :param img_width: width of the input image to the network
        :param r: percentage of pixels to consider for loss (0.3 ==> lowest 30% confident predictions)
        :param ignore_index: class label of pixels which should be ignored in the los
        :param start_decay_epoch: last epoch with ratio = 1.0
        :param end_decay_epoch: first epoch with ratio = r
        :param weights: tensor of length num_classes with one class weight for each class.
        :param
        """
        super(BootstrappedCrossEntropy, self).__init__()

        self.img_w = img_width
        self.img_h = img_height
        self.ignore_index = ignore_index

        self.start_decay_epoch = start_decay_epoch
        self.end_decay_epoch = end_decay_epoch

        if self.start_decay_epoch is None and self.end_decay_epoch is None:
            # no decay always keep the same ratio of r
            self.ratio = r
            self.k = int(r * img_height * img_width)

        elif self.start_decay_epoch is not None and self.end_decay_epoch is not None:
            # decay ratio from start_decay_epoch with ratio = 1.0 to end_decay_ratio with ratio = r
            assert self.start_decay_epoch < self.end_decay_epoch
            self.ratio = None  # will be set in forward
            self.k = None  # will be set in forward
            diff_epochs = (end_decay_epoch - start_decay_epoch)
            ratio_step = (1.0 - r) / diff_epochs
            self.ratio_list = [1.0 - (i * ratio_step) for i in range(diff_epochs + 1)]
            self.k_list = [int(rat * img_height * img_width) for rat in self.ratio_list]

        else:
            raise ValueError('Both start_decay_epoch and end_decay_eopch have to be set, not only one of them!')

    def forward(self, prediction, target, epoch=None):
        """
        Calculates the bootstrapped cross entropy of the lowest k predictions.
        :param prediction: Tensor of shape [minibatch, n_classes, height, width]
        :param target: Tensor of shape [minibatch, height, width]
        :param epoch: current epoch, only necessary if using decaying ratio
        :return:
        """
        if self.start_decay_epoch is not None and self.end_decay_epoch is not None:
            # when using decaying ratio, we need to pass an epoch
            assert epoch is not None, \
                'Bootstrapped Cross Entropy is in decay mode, please pass current epoch to forward call'
            if epoch <= self.start_decay_epoch:
                # epoch befor decay start
                self.ratio = self.ratio_list[0]
                self.k = self.k_list[0]
            elif epoch < self.end_decay_epoch:
                # epoch with decaying ratio
                self.ratio = self.ratio_list[epoch - self.start_decay_epoch]
                self.k = self.k_list[epoch - self.start_decay_epoch]
            else:
                # decay already reached goal
                self.ratio = self.ratio_list[-1]
                self.k = self.k_list[-1]

        pre_one_hot = torch.clone(target).transpose(1, 2)

        # mask with true for each pixel and class probability prediction
        # false for each class probability prediction of pixel labeled ignore_index
        mask_loss = (pre_one_hot != self.ignore_index).transpose(1, 2).expand(*prediction.shape)

        # mask with true for gt class of each non ignore_index pixel, all other value are false
        pre_one_hot[pre_one_hot == self.ignore_index] = 0
        mask_class = torch.nn.functional.one_hot(pre_one_hot,
                                                 num_classes=prediction.shape[1])
        mask_class = mask_class.transpose(1, 3) == 1

        # class probability prediction of only ground truth classes of non ignore index pixels
        mask_class = torch.logical_and(mask_loss, mask_class)

        loss = - torch.log(prediction[mask_class] + 1e-9)

        if self.k > loss.shape[0]:
            self.k = loss.shape[0]

        # select the losses of the lowest k predictions (the same as the highest k losses)
        values, _ = torch.topk(loss, k=self.k, largest=True, sorted=False, dim=0)
        loss = torch.sum(values) / self.k
        return loss


class SurfaceNormalRegularizationLoss(nn.Module):
    """
    Surface normal regularization loss as defined in the paper
    "Geometric Unsupervised Domain Adaptation for Semantic Segmentation": https://arxiv.org/pdf/2103.16694.pdf
    """

    def __init__(self, ref_img_width, ref_img_height, normalized_camera_model, device):
        super(SurfaceNormalRegularizationLoss, self).__init__()
        self.device = device
        self.camera_model = normalized_camera_model.get_scaled_model(ref_img_width, ref_img_height)
        self.similiarity_loss = F.cosine_similarity
        self.img_to_pointcloud = _ImageToPointcloud(camera_model=self.camera_model, device=device)
        self.image_width, self.image_height = self.camera_model.image_size()
        self.cos_sim = nn.CosineSimilarity(dim=1)
        self.padder_h = torch.nn.ReflectionPad2d((0, 0, 0, 1))
        self.padder_w = torch.nn.ReflectionPad2d((0, 1, 0, 0))

    def get_normal_vectors(self, points3d):
        """
        Estimates the surface normal from 3D points as stated in https://arxiv.org/pdf/2103.16694.pdf
        :param points3d: tensor of shape [batch_size, 3, image_height, image_width]
        :return:
        """
        diff_h = (points3d[:, :, 0:, :] - self.padder_h(points3d[:, :, 1:, :]))
        diff_w = (points3d[:, :, :, 0:] - self.padder_w(points3d[:, :, :, 1:]))

        normals = torch.cross(diff_w, diff_h, dim=1)
        vector_norms = torch.unsqueeze(torch.linalg.vector_norm(normals, dim=1), dim=1) + 1e-09

        return torch.divide(normals, vector_norms)

    def cosine_similarity_guda(self, normals_pred, normals_gt):
        """
        Calculates the cosine similarity as stated in https://arxiv.org/pdf/2103.16694.pdf
        :param normals_pred: normals calculated from predicted depth
        :param normals_gt: normals calculated from ground truth depth
        """
        return torch.mean(torch.divide(torch.sum(1 - self.cos_sim(normals_pred, normals_gt), dim=(1, 2))
                                       , 2 * self.image_height * self.image_width))

    def forward(self, depth_prediction, depth_gt):
        """
        :param depth_prediction: predicted depth (not inverse depth!)
        :param depth_gt: ground truth depth (not inverse depth!)
        :return:
        """
        points3d_prediction = self.img_to_pointcloud(depth_prediction)
        points3d_gt = self.img_to_pointcloud(depth_gt)

        normals_prediction = self.get_normal_vectors(points3d_prediction)
        normals_gt = self.get_normal_vectors(points3d_gt)

        return self.cosine_similarity_guda(normals_pred=normals_prediction, normals_gt=normals_gt)


class SemanticL1LossPixelwise(nn.Module):
    """
    Implements the absolute difference between prediction and target per pixel.
    """

    def __init__(self):
        super(SemanticL1LossPixelwise, self).__init__()

    def forward(self, nonwarped_img_semantic, warped_img_semantic):
        mask = (warped_img_semantic == IGNORE_INDEX_SEMANTIC).detach()
        assert nonwarped_img_semantic.dim() == warped_img_semantic.dim(), "Inconsistent dimensions in L1Loss between prediction and target"
        loss = (nonwarped_img_semantic - warped_img_semantic).abs()
        zeros = torch.zeros_like(loss).detach()
        loss[mask] = zeros[mask]
        return loss

# ----------------------------------------------Motion Losses-----------------------------------------------------------


class MotionGroupSmoothnessRegularizationLoss(nn.Module):
    def __init__(self):
        super(MotionGroupSmoothnessRegularizationLoss, self).__init__()

    def forward(self, translation_map, warp_around=True):
        """
        :param translation_map: predicted dense translation map
        :param warp_around: True if continuity of smoothness around object boundaries should be enforced (smooth across
        left and right boundary pixels, as well as top and bottom)
        :return: smooth translation tensor
        """
        tensor_dx = translation_map - torch.roll(translation_map, shifts=1, dims=3)
        tensor_dy = translation_map - torch.roll(translation_map, shifts=1, dims=2)
        if not warp_around:
            tensor_dx = tensor_dx[:, :, 1:, 1:]
            tensor_dy = tensor_dy[:, :, 1:, 1:]
        return torch.mean(torch.sqrt(1e-24 + torch.square(tensor_dx) + torch.square(tensor_dy)))


class MotionSparsityRegularizationLoss(nn.Module):
    def __init__(self):
        super(MotionSparsityRegularizationLoss, self).__init__()

    def forward(self, translation_map):
        """
        :param translation_map: predicted dense translation map
        :return: smooth translation tensor
        """
        abs_map = torch.abs(translation_map)
        spatial_mean_motion = torch.mean(abs_map, dim=(2, 3), keepdim=True).detach()
        print(spatial_mean_motion)
        return torch.mean(2 * spatial_mean_motion * torch.sqrt(abs_map/(spatial_mean_motion + 1e-24) + 1))
# ----------------------------------------------Depth Losses------------------------------------------------------------


class L1LossDepth(nn.Module):
    """
    Implements the average over pixel of absolute difference between prediction and target.

    This can be used for both sparse and dense depth supervision. Sparse: Set the depth values for those pixels with
    no measurements to zero.
    """

    def __init__(self):
        super(L1LossDepth, self).__init__()

    def forward(self, prediction, target):
        assert prediction.dim() == target.dim(), "Inconsistent dimensions in L1Loss between prediction and target"
        mask = (target > 0).detach()
        abs_diff = (prediction - target)[mask].abs()
        loss = abs_diff.mean()
        return loss


class SilogLossDepth(nn.Module):
    """
    Implementation of the Scale Invariant Log Loss introduced in
    'Depth map prediction from a single image using a multi-scale deep network'.
    """

    def __init__(self, weight=0.85, ignore_value=-100):
        super(SilogLossDepth, self).__init__()
        self.weight = weight
        self.ignore_value = ignore_value

    def forward(self, pred, target):
        mask = (torch.logical_and(target > 0, target != self.ignore_value)).detach()
        log_diff = (torch.log(pred) - torch.log(target))[mask]

        silog_var = torch.mean(log_diff ** 2)
        silog_wsme = self.weight * (log_diff.mean() ** 2)

        return silog_var - silog_wsme


class L1LossPixelwise(nn.Module):
    """
    Implements the absolute difference between prediction and target per pixel.
    """

    def __init__(self):
        super(L1LossPixelwise, self).__init__()

    def forward(self, prediction, target):
        assert prediction.dim() == target.dim(), "Inconsistent dimensions in L1Loss between prediction and target"
        loss = (prediction - target).abs()
        return loss


class MSELoss(nn.Module):
    """
    Implements the mean squared difference between prediction and target.

    This can be used for both sparse and dense depth supervision. Sparse: Set the depth values for those pixels with
    no measurements to zero.
    """

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, prediction, target):
        assert prediction.dim() == target.dim(), "Inconsistent dimensions in MSELoss between prediction and target"
        mask = (target > 0).detach()
        abs_squared_diff = torch.square((prediction - target)[mask])
        loss = abs_squared_diff.mean()
        return loss


class EdgeAwareSmoothnessLoss(nn.Module):
    """
    The edge aware smoothness Loss enforces smooth depth gradients in regions where no edges are detected.
    """

    def __init__(self):
        super(EdgeAwareSmoothnessLoss, self).__init__()

    def forward(self, inv_depth_map, image):
        # Compute the gradients
        abs_grad_inv_depth_x = (inv_depth_map[:, :, :, 1:] - inv_depth_map[:, :, :, :-1]).abs()
        abs_grad_inv_depth_y = (inv_depth_map[:, :, 1:, :] - inv_depth_map[:, :, :-1, :]).abs()

        abs_grad_image_x = ((image[:, :, :, 1:] - image[:, :, :, :-1]).abs()).mean(1, keepdim=True)
        abs_grad_image_y = ((image[:, :, 1:, :] - image[:, :, :-1, :]).abs()).mean(1, keepdim=True)

        # Compute the final loss
        loss_x = abs_grad_inv_depth_x * torch.exp(-abs_grad_image_x)
        loss_y = abs_grad_inv_depth_y * torch.exp(-abs_grad_image_y)

        loss = loss_x.mean() + loss_y.mean()

        return loss


class SSIMLoss(nn.Module):
    """
    Implements the structural similarity component from
    'Image quality assessment: from error visibility to structural similarity'
    """

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
    """
    Implements the reconstruction Loss from
    'Loss functions for image restoration with neural networks'
    with optional SSIM loss and minimum error selection from
    ' Unsupervised learning of depth and ego-motion from video'
    """

    # Attention, we use a camera model here, that is normalized by the image size, in order to be able to compute
    # reprojections for each possible size of the image
    # ref_img_width and ref_img_height denote the original image sizes without scalings
    def __init__(self, ref_img_width, ref_img_height, normalized_camera_model, num_scales, device, ssim_kernel_size=3):
        super(ReconstructionLoss, self).__init__()
        self.num_scales = num_scales
        self.l1_loss_pixelwise = L1LossPixelwise()
        self.semantic_l1_loss_pixelwise = SemanticL1LossPixelwise()
        self.ssim = SSIMLoss(ssim_kernel_size)
        self.image_warpers = {}
        self.scaling_modules = {}

        # Get camera models that are adapted to the actual (scaled) image sizes
        for i in range(self.num_scales):
            s = 2 ** i
            scale_u = ref_img_width // s
            scale_v = ref_img_height // s
            camera_model_ = normalized_camera_model.get_scaled_model(scale_u, scale_v)
            self.image_warpers[i] = ImageWarper(camera_model_, device)
            self.scaling_modules[i] = torch.nn.AvgPool2d(kernel_size=s, stride=s)

    def forward(self, batch_data, pred_depth, poses, rgb_frame_offsets, use_automasking, use_ssim, alpha=0.85,
                mask=None, semantic_logits=None, motion_map=None):
        """
        :param semantic_logits: dictionary of semantic softmax predictions
        :param batch_data: batch of data containing the rgb images
        :param pred_depth: batch of predicted depths for the middle frames
        :param poses: batch of predicted poses
        :param rgb_frame_offsets: offsets of the rgb images to the middle frame
        :param use_automasking: auto mask stationary pixels
        :param use_ssim: if ssim compontent of loss should be used
        :param alpha: weighting between both losses
        :param mask:
        :param batch_semantic_logits: batch of semantic softmax predictions
        :return:
        """
        assert rgb_frame_offsets[
                   0] == 0  # This should always be zero to make sure that the first id corresponds
        # to the current rgb image the depth is predicted for

        total_loss = 0.0

        semantic_cosistency_losses = []  # init list here unlike other losses,
        # since semantic only evaluated at largest scale

        # Scale down the depth image (scaling must be part of the optimization graph!)

        semantic_cosistency_loss_min = None
        for s in range(self.num_scales):

            reconstruction_losses_s = []
            identity_losses_s = []

            scaled_depth = pred_depth[('depth', s)]
            scaled_tgt_img = self.scaling_modules[s](batch_data[("rgb", 0)])

            for frame_id in rgb_frame_offsets[1:]:
                if motion_map is not None:
                    scaled_motion_map = self.scaling_modules[s](motion_map[frame_id])
                else:
                    scaled_motion_map = None

                scaled_adjacent_img_ = self.match_sizes(batch_data[("rgb", frame_id)], scaled_depth.shape)

                if s == 0 and semantic_logits is not None:
                    # compare semantic predictions (if available) only on the original input resolution
                    warped_scaled_adjacent_img_, warped_semantic_logits_frame = \
                        self.image_warpers[s](scaled_adjacent_img_,
                                              scaled_depth,
                                              poses[frame_id],
                                              semantic_logits[frame_id],
                                              scaled_motion_map)

                    loss_sem = self.semantic_l1_loss_pixelwise(semantic_logits[0], warped_semantic_logits_frame)

                    semantic_cosistency_losses.append(loss_sem)

                    #plt.imshow(loss_sem.mean(1, True)[0, 0, :, :].cpu().detach())
                    #plt.show()

                else:
                    warped_scaled_adjacent_img_ = self.image_warpers[s](scaled_adjacent_img_,
                                                                        scaled_depth,
                                                                        poses[frame_id],
                                                                        None,
                                                                        scaled_motion_map)

                rec_l1_loss = self.l1_loss_pixelwise(warped_scaled_adjacent_img_, scaled_tgt_img).mean(1, True)
                rec_ssim_loss = self.ssim(warped_scaled_adjacent_img_, scaled_tgt_img).mean(1, True)

                # Compute pixelwise reconstruction loss by combining photom. loss and ssim loss
                reconstruction_losses_s.append(
                    (1 - alpha) * rec_l1_loss + alpha * rec_ssim_loss if use_ssim else rec_l1_loss)

                if use_automasking:
                    id_l1_loss = self.l1_loss_pixelwise(scaled_adjacent_img_, scaled_tgt_img).mean(1, True)
                    id_ssim_loss = self.ssim(scaled_adjacent_img_, scaled_tgt_img).mean(1, True)
                    identity_loss = (1 - alpha) * id_l1_loss + alpha * id_ssim_loss if use_ssim else id_l1_loss
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

            total_loss += final_loss_s / (2 ** s)

            # calculate the semantic consistency loss if semantic prediction is given
            if s == 0 and semantic_logits is not None:
                # the warped image index pixels that have minimum loss in photometric reconstruction
                image_pixels_used = torch.argmin(reconstruction_losses_s,
                                                 dim=1).repeat(1, semantic_cosistency_losses[0].shape[1], 1, 1)

                # take the loss of the pixels which had minimum loss in photometric reconstruction
                semantic_cosistency_loss_min = torch.zeros_like(semantic_cosistency_losses[0])
                for i, loss_tensor in enumerate(semantic_cosistency_losses):
                    semantic_cosistency_loss_min[image_pixels_used == i] = loss_tensor[image_pixels_used == i]

                semantic_cosistency_loss_min = semantic_cosistency_loss_min.mean()

        if semantic_logits is None:
            # no semantic consistency calculated
            return total_loss, 0
        else:
            return total_loss, semantic_cosistency_loss_min

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

        return self.l1_loss(coordinates_pred, coordinates_gt)  # ToDo: Check this

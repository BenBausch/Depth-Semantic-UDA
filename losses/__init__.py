from losses.losses import *


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

loss_dict = {
    'reconstruction': ReconstructionLoss,
    'ssim': SSIMLoss,
    'edge_smooth': EdgeAwareSmoothnessLoss,
    'mse': MSELoss,
    'l1_depth': L1LossDepth,
    'silog_depth': SilogLossDepth,
    'depth_reprojection': DepthReprojectionLoss,
    'l1_pixelwise': L1LossPixelwise,
    'cross_entropy': nn.CrossEntropyLoss,
    'weighted_cross_entropy': WeightedCrossEntropy,
    'bootstrapped_cross_entropy': BootstrappedCrossEntropy,
    'surface_normal_regularization': SurfaceNormalRegularizationLoss
}
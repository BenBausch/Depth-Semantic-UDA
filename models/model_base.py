import abc

import torch.nn as nn

class ModelBase(nn.Module, abc.ABC):
    """
    Base model for depth completion/monocular depth prediction tasks
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, depth_network=None, pose_network=None):
        super(ModelBase, self).__init__()

    @abc.abstractmethod
    def predict_depth(self, imgs):
        pass

    @abc.abstractmethod
    def predict_poses(self, images):
        pass

    @abc.abstractmethod
    def forward(self, batch):
        pass

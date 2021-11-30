import abc
from abc import ABC

import torch.nn as nn


class ModelBase(nn.Module, abc.ABC):
    """
    Base model for depth completion/monocular depth prediction tasks
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, depth_network=None, pose_network=None):
        super(ModelBase, self).__init__()

    @abc.abstractmethod
    def predict_semantic(self, features):
        pass

    @abc.abstractmethod
    def predict_depth(self, features):
        pass

    @abc.abstractmethod
    def predict_poses(self, images):
        pass

    @abc.abstractmethod
    def forward(self, batch):
        pass

    @abc.abstractmethod
    def latent_features(self, images):
        pass


class DepthModelBase(ModelBase, ABC):
    def __init__(self, depth_network, pose_network):
        super(DepthModelBase, self).__init__(depth_network=depth_network, pose_network=pose_network)

    def predict_semantic(self, features):
        return None


class SemanticModelBase(ModelBase, ABC):
    def __init__(self, depth_network, pose_network):
        super(DepthModelBase, self).__init__(depth_network=depth_network, pose_network=pose_network)

    def predict_depth(self, features):
        return None
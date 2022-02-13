import abc

import torch.nn as nn


class EncoderDecoderModelBase(nn.Module, abc.ABC):
    """
    Base model incorporating all functions that depth and semantic models need to have!
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(EncoderDecoderModelBase, self).__init__()

    @abc.abstractmethod
    def create_Encoder(self):
        """
        Creates the encoder resnet.
        """
        pass

    @abc.abstractmethod
    def forward(self, batch):
        """
        Implements the forward pass trough all the networks.
        :param batch: batch of images.
        """
        pass

    @abc.abstractmethod
    def latent_features(self, images):
        """
        Predicts the latent embedding of the input images.
        :param images: batch of images.
        """
        pass


class DepthFromMotionEncoderDecoderModelBase(EncoderDecoderModelBase, abc.ABC):
    """
    Extension of the Base model indicating all functions a monocular depth model is obligated to have!
    """
    def __init__(self):
        super(DepthFromMotionEncoderDecoderModelBase, self).__init__()

    @abc.abstractmethod
    def create_PoseNet(self):
        """
        Creates the network (encoder and decoder) of the PoseNet.
        """
        pass

    @abc.abstractmethod
    def create_DepthNet(self):
        """
        Creates the network for the depth head.
        """
        pass

    @abc.abstractmethod
    def predict_depth(self, features):
        """
        Predicts the depth by executing a forward pass of latent features trough the depth head.
        :param features: latent feature embedding received from the encoder.
        """
        pass

    @abc.abstractmethod
    def predict_poses(self, images):
        """
        Predicts the 6 degrees of freedom translation parameters between the views.
        :param images: multiple input images of the same scene.
        """
        pass

    @staticmethod
    def get_number_of_posenet_input_frames(cfg):
        """
        Retrieves the number of input frames to the PoseNet from the configuration
        :param cfg: configuration
        """
        if cfg.model.pose_net.input == 'pairs':
            return 2
        elif cfg.model.pose_net.input == 'all':
            return cfg.train.rgb_frame_offsets
        else:
            raise Exception('Invalid offset pairs for poseNet! Choose: \'pairs\' for pairs of 2, or ' +
                            '\'all\' for all the offset images!')


class SemanticEncoderDecoderModelBase(EncoderDecoderModelBase, abc.ABC):
    """
    Extension of the Base model indicating all functions a semantic model is obligated to have!
    """
    def __init__(self):
        super(SemanticEncoderDecoderModelBase, self).__init__()

    @abc.abstractmethod
    def create_SemanticNet(self):
        """
        Creates the network for the semantic head.
        """
        pass

    @abc.abstractmethod
    def predict_semantic(self, features, *args):
        """
        Predicts the semantic segmentation of the input image ([batch_size, number_classes, height, width]) from the
        latent feature embedding.
        :param features: latent feature embedding of the input images received from the encoder.
        """
        pass


class SemanticDepthFromMotionModelBase(DepthFromMotionEncoderDecoderModelBase, SemanticEncoderDecoderModelBase, abc.ABC):
    """
    Combining depth from motion models with semantic models.
    """
    def __init__(self):
        super(SemanticDepthFromMotionModelBase, self).__init__()


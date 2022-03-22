import matplotlib.pyplot
from torch.utils.data import DataLoader

import time

from models.base.model_base import SemanticEncoderDecoderModelBase
from models.helper_models.resnet_encoder import ResnetEncoder
from models.helper_models.depth_decoder import DepthDecoderMONODEPTH2
from models.helper_models.semantic_decoder import SemanticDecoderGUDA
from models.helper_models.pose_decoder import PoseDecoder
from models.helper_models.layers import *


class PC2SModel(SemanticEncoderDecoderModelBase):

    # --------------------------------------------------------------------------
    # ------------------------Initialization-Methods----------------------------
    # --------------------------------------------------------------------------
    def __init__(self, cfg):
        """
        Constructor for guda network.
        :param cfg: the config file
        :param dataset: 0 if only one dataset used
                        0 if source dataset parameters shall be considered in case of 2 datasets
                        1 if source dataset parameters shall be considered in case of 2 datasets
        """
        super(PC2SModel, self).__init__()

        self.cfg = cfg

        self.num_layers_encoder = cfg.model.encoder.params.nof_layers  # which resnet to use
        # e.g. num_layers_encoder = 101 --> resnet101
        self.weights_init_encoder = cfg.model.encoder.params.weights_init
        self.no_cuda = cfg.device.no_cuda
        self.multiple_gpus = cfg.device.multiple_gpus

        self.num_classes = None  # initialized in get_dataset_parameters

        self.get_dataset_parameters()

        self.networks = nn.ModuleDict()
        self.parameters_to_train = []

        # create and add the different parts of the depth and segmentation network.
        self.create_Encoder()  # has to be called first before Depth and Semantic
        self.create_SemanticNet()

    def get_dataset_parameters(self):
        """
        Collects the parameters per dataset.
        :param dataset:
        :return:
        """

        # depending on the dataset one might only want to predict semantic masks or depth or depth with pose
        self.dataset_predict_semantic = []

        # all datasets should have the same number of classes because they use the same semantic head
        self.num_classes = self.cfg.datasets.configs[0].dataset.num_classes

        # collect the parameters from all the datasets
        for i, dcfg in enumerate(self.cfg.datasets.configs):
            # This network should predict semantic map for each dataset
            self.dataset_predict_semantic.append(True)

            if self.num_classes != dcfg.dataset.num_classes:
                raise ValueError('All datasets need to have same number of classes!')
        print(f'Predict semantic mask per dataset: {self.dataset_predict_semantic}')
        print(f'Number classes: {self.num_classes}')

    def create_Encoder(self):
        self.networks["resnet_encoder"] = ResnetEncoder(
            self.num_layers_encoder, self.weights_init_encoder == "pretrained").double()

        self.parameters_to_train += list(self.networks["resnet_encoder"].parameters())

    def create_SemanticNet(self):
        self.networks["semantic_decoder"] = SemanticDecoderGUDA(
            self.networks["resnet_encoder"].num_ch_enc, self.num_classes, upsample_mode='bilinear').double()

        self.parameters_to_train += list(self.networks["semantic_decoder"].parameters())

    # --------------------------------------------------------------------------
    # ---------------------------Prediction-Methods-----------------------------
    # --------------------------------------------------------------------------

    def latent_features(self, images):
        """Passes the images through the encoder to get the latent features"""
        return self.networks["resnet_encoder"](images)

    def predict_semantic(self, features):
        """Predict semantics using the latent features"""
        a = self.networks["semantic_decoder"](features)
        return a

    def forward(self, data):
        """
        Performs multiple forward passes, one for each dataset. This is need because of the distributed data parallel
        training, where a model is allowed to perform only a single forward pass before backward pass in order to keep
        the gradients synced across the gpus (DDP Hooks into the forward passes)
        :param data: list of point cloud batches
        :returns list of dictionary results, one for each dataset
        """
        all_results = []
        for dataset_id, batch in enumerate(data):
            all_results.append(self.single_forward(batch))
        return all_results

    def single_forward(self, batch):
        """
        Forward pass on a single batch.
        param batch: batch of data to process
        validation for visualization purposes) Note: Don't use to manage depth prediction per dataset -->
        set use_..._depth to True in the config of that specific dataset!
        """
        latent_features_batch = self.latent_features(batch)

        results = {'semantic': self.predict_semantic(latent_features_batch)}

        return results

    # --------------------------------------------------------------------------
    # -----------------------------Helper-Methods-------------------------------
    # --------------------------------------------------------------------------

    def params_to_train(self, *args):
        """
        Get all the trainable parameters.
        """
        return self.parameters_to_train

    def semantic_net(self):
        """
        Get dictionary of all the networks used for predicting the semantic segmentation.
        """
        return {k: self.networks[k] for k in ["resnet_encoder", "semantic_decoder"]}

    def get_networks(self):
        """
        Get all the networks of the model.
        """
        return self.networks

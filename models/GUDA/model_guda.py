import matplotlib.pyplot
from torch.utils.data import DataLoader

from models.base.model_base import SemanticDepthFromMotionModelBase
from models.helper_models.resnet_encoder import ResnetEncoder
from models.helper_models.depth_decoder import DepthDecoder
from models.helper_models.semantic_decoder import SemanticDecoder
from models.helper_models.pose_decoder import PoseDecoder

from models.helper_models.layers import *


class Guda(SemanticDepthFromMotionModelBase):

    # --------------------------------------------------------------------------
    # ------------------------Initialization-Methods----------------------------
    # --------------------------------------------------------------------------
    def __init__(self, device, cfg):
        """
        Constructor for guda network.
        :param device: device on which to run the network
        :param cfg: the config file
        :param dataset: 0 if only one dataset used
                        0 if source dataset parameters shall be considered in case of 2 datasets
                        1 if source dataset parameters shall be considered in case of 2 datasets
        """
        super(Guda, self).__init__()

        self.cfg = cfg

        self.num_layers_encoder = cfg.model.encoder.params.nof_layers  # which resnet to use
        # e.g. num_layers_encoder = 101 --> resnet101
        self.num_layers_pose = cfg.model.pose_net.params.nof_layers
        self.weights_init_encoder = cfg.model.encoder.params.weights_init
        self.weights_init_pose_net_encoder = cfg.model.pose_net.params.weights_init
        self.num_scales = cfg.model.depth_net.nof_scales
        self.no_cuda = cfg.device.no_cuda
        self.multiple_gpus = cfg.device.multiple_gpus
        self.pose_model_input = cfg.model.pose_net.input

        self.num_classes = None  # initialized in get_dataset_parameters
        self.rgb_frame_offsets = None  # initialized in get_dataset_parameters

        self.get_dataset_parameters()

        self.device = device

        self.networks = {}
        self.parameters_to_train = []

        # create and add the different parts of the depth and segmentation network.
        self.create_Encoder()  # has to be called first before Depth and Semantic
        self.create_DepthNet()
        self.create_SemanticNet()

        # create and set the attributes and network of PoseNet
        self.num_pose_frames = self.get_number_of_posenet_input_frames(cfg)

        self.create_PoseNet()

        print(f'Device: {self.device}')
        print('After creating networks')

    def get_dataset_parameters(self):  # todo move this into the base class
        """
        Collects the parameters per dataset.
        :param dataset:
        :return:
        """
        # rgb frame offsets vary across datasets if self-supervised depth is used or not.
        self.rgb_frame_offsets = []

        # depending on the dataset one might only want to predict semantic masks or depth or depth with pose
        self.dataset_predict_depth = []
        self.dataset_predict_semantic = []
        self.dataset_predict_pose = []
        self.dataset_min_max_depth = []

        # all datasets should have the same number of classes because they use the same semantic head
        self.num_classes = self.cfg.datasets.configs[0].dataset.num_classes

        # collect the parameters from all the datasets
        for i, dcfg in enumerate(self.cfg.datasets.configs):
            self.rgb_frame_offsets.append(dcfg.dataset.rgb_frame_offsets)

            # get the depth ranges for each dataset
            self.dataset_min_max_depth.append([dcfg.dataset.min_depth, dcfg.dataset.max_depth])

            # in all these cases depth prediction is needed
            self.dataset_predict_depth.append(dcfg.dataset.use_sparse_depth or
                                              dcfg.dataset.use_dense_depth or
                                              dcfg.dataset.use_self_supervised_depth)

            # pose prediction is only needed in context of self-supervised monocular depth
            self.dataset_predict_pose.append(dcfg.dataset.use_self_supervised_depth)

            self.dataset_predict_semantic.append(dcfg.dataset.use_semantic_gt)

            if self.num_classes != dcfg.dataset.num_classes:
                raise ValueError('All datasets need to have same number of classes!')
        print(f'Frame offsets for all datasets: {self.rgb_frame_offsets}')
        print(f'Predict depth per dataset: {self.dataset_predict_depth}')
        print(f'Predict pose per dataset: {self.dataset_predict_pose}')
        print(f'Predict semantic mask per dataset: {self.dataset_predict_semantic}')
        print(f'Number classes: {self.num_classes}')

    def create_Encoder(self):
        self.networks["resnet_encoder"] = ResnetEncoder(
            self.num_layers_encoder, self.weights_init_encoder == "pretrained").double()

        if not self.no_cuda and self.multiple_gpus:
            self.networks["resnet_encoder"] = \
                torch.nn.DataParallel(self.networks["resnet_encoder"],
                                      device_ids=[i for i in range(torch.cuda.device_count())]).cuda()
        else:
            self.networks["resnet_encoder"].to(self.device)

        self.parameters_to_train += list(self.networks["resnet_encoder"].parameters())

    def create_DepthNet(self):
        self.networks["depth_decoder"] = DepthDecoder(
            self.networks["resnet_encoder"].num_ch_enc, range(self.num_scales), upsample_mode='bilinear').double()

        if not self.no_cuda and self.multiple_gpus:
            self.networks["depth_decoder"] = \
                torch.nn.DataParallel(self.networks["depth_decoder"],
                                      device_ids=[i for i in range(torch.cuda.device_count())]).cuda()
        else:
            self.networks["depth_decoder"].to(self.device)

        self.parameters_to_train += list(self.networks["depth_decoder"].parameters())

    def create_SemanticNet(self):
        self.networks["semantic_decoder"] = SemanticDecoder(
            self.networks["resnet_encoder"].num_ch_enc, self.num_classes, upsample_mode='bilinear').double()

        if not self.no_cuda and self.multiple_gpus:
            self.networks["semantic_decoder"] = \
                torch.nn.DataParallel(self.networks["semantic_decoder"],
                                      device_ids=[i for i in range(torch.cuda.device_count())]).cuda()
        else:
            self.networks["semantic_decoder"].to(self.device)

        self.parameters_to_train += list(self.networks["semantic_decoder"].parameters())

    def create_PoseNet(self):
        # Specify pose encoder and decoder
        self.networks["pose_encoder"] = ResnetEncoder(
            self.num_layers_pose,
            self.weights_init_pose_net_encoder == "pretrained",
            num_input_images=self.num_pose_frames,
            wanted_scales=[1, 2, 3]).double()

        self.networks["pose_decoder"] = PoseDecoder(
            self.networks["pose_encoder"].get_channels_of_forward_features(),
            num_input_features=1,
            num_frames_to_predict_for=self.num_pose_frames).double()

        # Specify parameters to be trained
        self.parameters_to_train += list(self.networks["pose_encoder"].parameters())
        self.parameters_to_train += list(self.networks["pose_decoder"].parameters())

        # Specify device(s) to be used for training
        if not self.no_cuda and self.multiple_gpus:
            self.networks["pose_encoder"] = \
                torch.nn.DataParallel(self.networks["pose_encoder"],
                                      device_ids=[i for i in range(torch.cuda.device_count())]).cuda()
            self.networks["pose_decoder"] = \
                torch.nn.DataParallel(self.networks["pose_decoder"],
                                      device_ids=[i for i in range(torch.cuda.device_count())]).cuda()
        else:
            self.networks["pose_encoder"].to(self.device)
            self.networks["pose_decoder"].to(self.device)

    # --------------------------------------------------------------------------
    # ---------------------------Prediction-Methods-----------------------------
    # --------------------------------------------------------------------------

    def latent_features(self, images):
        return self.networks["resnet_encoder"](images)

    def predict_depth(self, features, dataset_id):
        """
        Predicts inverse depth map.
        :param features: features from the encoder.
        :param dataset_id: the dataset for which to predict the depth
        :return:
        """
        # The network actually outputs the inverse depth!
        raw_sigmoid = self.networks["depth_decoder"](features)
        raw_sigmoid_scale_0 = raw_sigmoid[("disp", 0)]
        _, depth_pred = disp_to_depth(disp=raw_sigmoid_scale_0,
                                      min_depth=self.dataset_min_max_depth[dataset_id][0],
                                      max_depth=self.dataset_min_max_depth[dataset_id][1])

        return depth_pred, raw_sigmoid_scale_0

    def predict_poses(self, inputs, dataset_id):
        """
        Adapted from: https://github.com/nianticlabs/monodepth2/blob/master/trainer.py
        Predict poses between input frames for monocular sequences.
        Slightly modified by Ben Bausch
        """
        poses = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.
            pose_feats = {f_i: inputs["rgb", f_i] for f_i in self.rgb_frame_offsets[dataset_id]}

            for f_i in self.rgb_frame_offsets[dataset_id][1:]:
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                axisangle, translation = self.networks["pose_decoder"](
                    [self.networks["pose_encoder"](torch.cat(pose_inputs, 1))])

                # Invert the matrix if the frame id is negative

                poses[f_i] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            raise Exception('The Input to the GUDA PoseNet model are exactly 2 frames!')

        return poses

    def predict_semantic(self, features):
        a = self.networks["semantic_decoder"](features)

        return a

    def forward(self, batch, dataset_id, predict_depth=False):
        """
        :param batch: batch of data to process
        :param dataset_id: number of datasets in the list of datasets (source: 0, target:1, ...)
        :param predict_depth: used to manually enforce depth prediction (for example: during specific samples in
        validation for visualization purposes) Note: Don't use to manage depth prediction per dataset -->
        set use_..._depth to True in the config of that specific dataset!
        """
        latent_features_batch = self.latent_features(batch[("rgb", 0)])

        results = {}

        if self.dataset_predict_depth[dataset_id] or predict_depth:
            results['depth'] = self.predict_depth(latent_features_batch, dataset_id=dataset_id)
        else:
            results['depth'] = None

        if self.dataset_predict_pose[dataset_id]:
            results['poses'] = self.predict_poses(batch, dataset_id=dataset_id)
        else:
            results['poses'] = None

        if self.dataset_predict_semantic[dataset_id]:
            results['semantic'] = self.predict_semantic(latent_features_batch)
        else:
            results['semantic'] = None

        return results

    # --------------------------------------------------------------------------
    # -----------------------------Helper-Methods-------------------------------
    # --------------------------------------------------------------------------

    def params_to_train(self):
        """
        Get all the trainable parameters.
        """
        return self.parameters_to_train

    def depth_net(self):
        """
        Get dictionary of all the networks used for predicting the depth.
        """
        return {k: self.networks[k] for k in ["resnet_encoder", "depth_decoder"]}

    def semantic_net(self):
        """
        Get dictionary of all the networks used for predicting the semantic segmentation.
        """
        return {k: self.networks[k] for k in ["resnet_encoder", "semantic_decoder"]}

    def pose_net(self):
        """
        Get dictionary of all the networks used for predicting the poses.
        """
        return {k: self.networks[k] for k in ["pose_encoder", "pose_decoder"]}

    def get_networks(self):
        """
        Get all the networks of the model.
        """
        return self.networks

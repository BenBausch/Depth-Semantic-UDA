from cfg.config_single_dataset import get_cfg_defaults
from dataloaders.dataset_gta5 import GTA5Dataset
from models.base.model_base import DepthFromMotionEncoderDecoderModelBase
from models.helper_models.resnet_encoder import ResnetEncoder
from models.helper_models.depth_decoder import DepthDecoder
from models.helper_models.pose_decoder import PoseDecoder

from models.helper_models.layers import *

# ToDo: Das Netz wird an die Anzahl der Skalen angepasst, um entsprechende Tiefenkarten auszugeben. Wir skalieren jedoch
#  die RGB Bilder und die ausgegebene Tiefenkarte herunter -> Die anzahl beider Skalen, d.h. die bezüglich der
#  Netzstruktur und die zur BErechnung des photom. Fehlers ist an self.num_scales gebunden -> FIX THAT


class Monodepth2(DepthFromMotionEncoderDecoderModelBase):

    # --------------------------------------------------------------------------
    # ------------------------Initialization-Methods----------------------------
    # --------------------------------------------------------------------------
    def __init__(self, device, cfg):
        super(Monodepth2, self).__init__()

        self.num_layers_encoder = cfg.model.encoder.params.nof_layers  # which resnet to use
        # e.g. num_layers_encoder = 101 --> resnet101
        self.num_layers_pose = cfg.model.pose_net.params.nof_layers
        self.weights_init_encoder = cfg.model.encoder.params.weights_init
        self.weights_init_pose_net_encoder = cfg.model.pose_net.params.weights_init
        self.num_scales = cfg.losses.reconstruction.nof_scales
        self.no_cuda = cfg.device.no_cuda
        self.multiple_gpus = cfg.device.multiple_gpus
        self.rgb_frame_offsets = cfg.train.rgb_frame_offsets
        self.pose_model_input = cfg.model.pose_net.input

        self.device = device

        self.networks = {}
        self.parameters_to_train = []

        # create and add the different parts of the depth and segmentation network.
        self.create_Encoder()  # has to be called first before Depth and Semantic
        self.create_DepthNet()

        # create and set the attributes and network of PoseNet
        self.num_pose_frames = self.get_number_of_posenet_input_frames(cfg)

        self.create_PoseNet()

    def create_Encoder(self):
        self.networks["resnet_encoder"] = ResnetEncoder(
            self.num_layers_encoder, self.weights_init_encoder == "pretrained").double()

        if not self.no_cuda and self.multiple_gpus:
            self.networks["resnet_encoder"] = torch.nn.DataParallel(self.networks["resnet_encoder"]).cuda()
        else:
            self.networks["resnet_encoder"].to(self.device)

        self.parameters_to_train += list(self.networks["resnet_encoder"].parameters())

    def create_DepthNet(self):
        self.networks["depth_decoder"] = DepthDecoder(
            self.networks["resnet_encoder"].num_ch_enc, range(self.num_scales), upsample_mode='nearest').double()

        if not self.no_cuda and self.multiple_gpus:
            self.networks["depth_decoder"] = torch.nn.DataParallel(self.networks["depth_decoder"]).cuda()
        else:
            self.networks["depth_decoder"].to(self.device)

        self.parameters_to_train += list(self.networks["depth_decoder"].parameters())

    def create_PoseNet(self):
        # Specify pose encoder and decoder
        self.networks["pose_encoder"] = ResnetEncoder(
            self.num_layers_pose,
            self.weights_init_pose_net_encoder == "pretrained",
            num_input_images=self.num_pose_frames).double()

        self.networks["pose_decoder"] = PoseDecoder(
            self.networks["pose_encoder"].get_channels_of_forward_features(),
            num_input_features=1,
            num_frames_to_predict_for=self.num_pose_frames).double()

        # Specify parameters to be trained
        self.parameters_to_train += list(self.networks["pose_encoder"].parameters())
        self.parameters_to_train += list(self.networks["pose_decoder"].parameters())

        # Specify device(s) to be used for training
        if not self.no_cuda and self.multiple_gpus:
            self.networks["pose_encoder"] = torch.nn.DataParallel(self.networks["pose_encoder"]).cuda()
            self.networks["pose_decoder"] = torch.nn.DataParallel(self.networks["pose_decoder"]).cuda()
        else:
            self.networks["pose_encoder"].to(self.device)
            self.networks["pose_decoder"].to(self.device)

    # --------------------------------------------------------------------------
    # ---------------------------Prediction-Methods-----------------------------
    # --------------------------------------------------------------------------

    def latent_features(self, images):
        return self.networks["resnet_encoder"](images)

    def predict_depth(self, features):
        # The network actually outputs the inverse depth!
        raw_sigmoid = self.networks["depth_decoder"](features)
        raw_sigmoid_scale_0 = raw_sigmoid[("disp", 0)]
        _, depth_pred = disp_to_depth(raw_sigmoid_scale_0)
        return depth_pred, raw_sigmoid_scale_0

    def predict_poses(self, inputs):
        """
        Adapted from: https://github.com/nianticlabs/monodepth2/blob/master/trainer.py
        Predict poses between input frames for monocular sequences.
        """
        poses = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.
            pose_feats = {f_i: inputs["rgb", f_i] for f_i in self.rgb_frame_offsets}

            for f_i in self.rgb_frame_offsets[1:]:
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [self.networks["pose_encoder"](torch.cat(pose_inputs, 1))]

                axisangle, translation = self.networks["pose_decoder"](pose_inputs)

                # Invert the matrix if the frame id is negative
                poses[f_i] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            pose_inputs = torch.cat([inputs[("rgb", i)] for i in self.opt.rgb_frame_offsets], 1)
            pose_inputs = [self.networks["pose_encoder"](pose_inputs)]

            axisangle, translation = self.networks["pose_decoder"](pose_inputs)

            for i, f_i in enumerate(self.opt.rgb_frame_offsets[1:]):
                poses[f_i] = transformation_from_parameters(axisangle[:, i], translation[:, i])

        return poses

    def forward(self, batch):
        latent_features_batch = self.latent_features(batch["rgb", 0])
        return {
            'depth': self.predict_depth(latent_features_batch),
            'poses': self.predict_poses(batch)
        }

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

if __name__=="__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file(
        r'C:\Users\benba\Documents\University\Masterarbeit\Depth-Semantic-UDA\cfg\train_gta5_semantic.yaml')
    cfg.eval.train.gt_depth_available = False
    cfg.eval.val.gt_depth_available = False
    cfg.eval.test.gt_depth_available = False
    cfg.dataset.use_sparse_depth = False
    cfg.eval.train.gt_semantic_available = True
    cfg.eval.val.gt_semantic_available = True
    cfg.eval.test.gt_semantic_available = True
    cfg.freeze()
    gta_dataset = GTA5Dataset('train', 'train', cfg)
    net = Monodepth2('cpu', cfg)

    print(net.networks['depth_encoder'].forward((next(iter(gta_dataset)))[('rgb', 0)].unsqueeze(0)))
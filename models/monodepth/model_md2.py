from models.model_base import DepthModelBase
from models.helper_models.resnet_encoder import ResnetEncoder
from models.helper_models.depth_decoder import DepthDecoder
from models.helper_models.pose_decoder import PoseDecoder

from train.layers import *

# ToDo: Das Netz wird an die Anzahl der Skalen angepasst, um entsprechende Tiefenkarten auszugeben. Wir skalieren jedoch
#  die RGB Bilder und die ausgegebene Tiefenkarte herunter -> Die anzahl beider Skalen, d.h. die bezüglich der
#  Netzstruktur und die zur BErechnung des photom. Fehlers ist an self.num_scales gebunden -> FIX THAT


class ModelMonodepth2(DepthModelBase):
    def __init__(self, device, cfg):
        super(ModelMonodepth2, self).__init__()

        self.num_layers = cfg.model.depth_net.params.nof_layers
        self.weights_init_depth_net_encoder = cfg.model.depth_net.params.weights_init
        self.weights_init_pose_net_encoder = cfg.model.pose_net.params.weights_init
        self.num_scales = cfg.losses.reconstruction.nof_scales
        self.no_cuda = cfg.device.no_cuda
        self.multiple_gpus = cfg.device.multiple_gpus
        self.rgb_frame_offsets = cfg.train.rgb_frame_offsets
        self.pose_model_input = cfg.model.pose_net.input

        self.networks = {}
        self.parameters_to_train = []

        # Specify depth network (Monodepth 2)
        self.networks["depth_encoder"] = ResnetEncoder(
            self.num_layers, self.weights_init_depth_net_encoder == "pretrained")

        self.networks["depth_decoder"] = DepthDecoder(
            self.networks["depth_encoder"].num_ch_enc, range(self.num_scales))

        if not self.no_cuda and self.multiple_gpus:
            self.networks["depth_encoder"] = torch.nn.DataParallel(self.networks["depth_encoder"]).cuda()
            self.networks["depth_decoder"] = torch.nn.DataParallel(self.networks["depth_decoder"]).cuda()
        else:
            self.networks["depth_encoder"].to(device)
            self.networks["depth_decoder"].to(device)

        self.parameters_to_train += list(self.networks["depth_encoder"].parameters())
        self.parameters_to_train += list(self.networks["depth_decoder"].parameters())

        # Specify pose network (Monodepth2)
        self.num_pose_frames = 2 if self.pose_model_input == "pairs" else self.num_input_frames

        # Specify pose encoder and decoder
        self.networks["pose_encoder"] = ResnetEncoder(
            self.num_layers,
            self.weights_init_pose_net_encoder == "pretrained",
            num_input_images=self.num_pose_frames)

        self.networks["pose_decoder"] = PoseDecoder(
            self.networks["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)

        # Specify parameters to be trained
        self.parameters_to_train += list(self.networks["pose_encoder"].parameters())
        self.parameters_to_train += list(self.networks["pose_decoder"].parameters())

        # Specify device(s) to be used for training
        if not self.no_cuda and self.multiple_gpus:
            self.networks["pose_encoder"] = torch.nn.DataParallel(self.networks["pose_encoder"]).cuda()
            self.networks["pose_decoder"] = torch.nn.DataParallel(self.networks["pose_decoder"]).cuda()
        else:
            self.networks["pose_encoder"].to(device)
            self.networks["pose_decoder"].to(device)

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
        latent_features_batch = self.latent_features(batch)
        return {
            'depth': self.predict_poses(latent_features_batch),
            'poses': self.predict_depth(batch["rgb", 0])
        }

    def params_to_train(self):
        return self.parameters_to_train

    def depth_net(self):
        return {k: self.networks[k] for k in ["depth_encoder", "depth_decoder"]}

    def pose_net(self):
        return {k: self.networks[k] for k in ["pose_encoder", "pose_decoder"]}

    def get_networks(self):
        return self.networks

    def latent_features(self, images):
        return self.networks["resnet_encoder"](images)

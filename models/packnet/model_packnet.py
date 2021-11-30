from models.model_base import DepthModelBase
from models.helper_models.resnet_encoder import ResnetEncoder
from models.helper_models.pose_decoder import PoseDecoder
from .PackNet01 import PackNet01

from train.layers import *

#TODO adapt this code to changes made by Ben

# ToDo: Die skalierten Ausgaben werden nicht im photometrischen Fehler berücksichtigt. Wir skalieren explizit nach...
# ToDo: The depth network should also use the sparse input for depth estimation -> need other model, e.g. self-sup


class ModelPacknet(DepthModelBase):
    def __init__(self, device, cfg):
        super(ModelPacknet, self).__init__()

        self.weights_init_pose_net_encoder = cfg.model.pose_net.params.weights_init
        # self.num_scales = opts.num_scales
        self.nof_layers = cfg.model.depth_net.params.nof_layers

        self.no_cuda = cfg.device.no_cuda
        self.multiple_gpus = cfg.device.multiple_gpus
        self.rgb_frame_offsets = cfg.train.rgb_frame_offsets
        self.pose_model_input = cfg.model.pose_net.input

        self.networks = {}
        self.parameters_to_train = []

        # Specify depth network (Packnet)
        self.networks["depth_net"] = PackNet01()

        if not self.no_cuda and self.multiple_gpus:
            self.networks["depth_net"] = torch.nn.DataParallel(self.networks["depth_net"]).cuda()
        else:
            self.networks["depth_net"].to(device)

        self.parameters_to_train += list(self.networks["depth_net"].parameters())

        # Specify pose network (Monodepth 2)
        self.num_pose_frames = 2 if self.pose_model_input == "pairs" else self.num_input_frames

        # Specify pose encoder and decoder
        self.networks["pose_encoder"] = ResnetEncoder(
            self.nof_layers,
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
        inv_depth = self.networks["depth_net"](features)
        return 1.0/inv_depth, None

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
        return {
            'depth': self.predict_poses(batch),
            'poses': self.predict_depth(batch["rgb", 0])
        }

    def params_to_train(self):
        return self.parameters_to_train

    def depth_net(self):
        return {k: self.networks[k] for k in ["depth_net"]}

    def pose_net(self):
        return {k: self.networks[k] for k in ["pose_encoder", "pose_decoder"]}

    def get_networks(self):
        return self.networks

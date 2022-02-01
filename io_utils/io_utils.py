import os
import json
import torch
from cfg.config_training import create_configuration


class IOHandler:
    """
    Class for handling model saving and model loading.
    """
    def __init__(self, path_base):
        self.path_base = path_base

    # ToDo: Specify the best model and save it as such
    def save_checkpoint(self, info, checkpoint):
        # Create folder
        path_save_folder = os.path.join(self.path_base, info["time"] + "_" + info["dataset"],  "checkpoints")

        if not os.path.exists(path_save_folder):
            os.makedirs(path_save_folder)

        # Create filename
        filename_save_file = "checkpoint_epoch_{}.pth".format(info["epoch"])

        # Save checkpoint
        path_save = os.path.join(path_save_folder, filename_save_file)
        print("Saving checkpoint for epoch {}".format(info["epoch"]))
        torch.save(checkpoint, path_save)

        # Give only read/execute permissions to everyone for the files created
        os.chmod(path_save, 0o555)

    def save_cfg(self, info, cfg):
        """
        Source: https://github.com/nianticlabs/monodepth2/blob/master/trainer.py
        Save options to disk so we know what we ran this experiment with
        """
        # Create folder
        path_save_folder = os.path.join(self.path_base, info["time"] + "_" + info["dataset"])

        # to_save = cfg.dump().__dict__.copy()

        if not os.path.exists(path_save_folder):
            os.makedirs(path_save_folder)

        to_save = dict(cfg)

        with open(os.path.join(path_save_folder, 'cfg.yaml'), 'w') as f:
            json.dump(to_save, f, indent=2)

    @staticmethod
    def load_cfg(path_cfg):
        cfg = create_configuration(path_cfg)
        return cfg

    @staticmethod
    def load_weights(checkpoint, models, optimizer=None):
        # Go through each model and update weights with those stored in the checkpoint
        for model_name, model in models.items():
            model.load_state_dict(checkpoint[model_name])
            print("Loading weights of {} on device {}".format(model_name, str(list(model.parameters())[0].device)))
            for key_item_1, key_item_2 in zip(model.state_dict().items(), checkpoint[model_name].items()):
                assert torch.equal(key_item_1[1], key_item_2[1]), \
                    "Mismatch found concerning model weights after loading!"
                assert key_item_1[0] == key_item_2[0], \
                    "Mismatch found concerning model keys after loading. Check whether you load the correct model."

        if optimizer is not None:
            # Load also the corresponding optimizer
            optimizer.load_state_dict(checkpoint["optimizer"])

        return checkpoint

    @staticmethod
    def gen_checkpoint(models, **kwargs):
        checkpoint = {k: val for k, val in kwargs.items()}
        models_dict = {k: val.state_dict() for k, val in models.items()}
        checkpoint.update(models_dict)
        return checkpoint

    @staticmethod
    def load_checkpoint(path_base_checkpoint, filename_checkpoint, device_id):
        path_checkpoint = os.path.join(path_base_checkpoint, 'checkpoints', filename_checkpoint)
        assert os.path.exists(path_checkpoint), "The path {} does not exist!".format(path_checkpoint)
        checkpoint = torch.load(path_checkpoint, map_location=torch.device(device_id))
        return checkpoint

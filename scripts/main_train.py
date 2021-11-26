# my_project/main.py

from train import train_depth
from cfg.config import get_cfg_defaults  # local variable usage pattern, or:
# from config import cfg  # global singleton usage pattern


if __name__ == "__main__":
    # ToDo: Take this in as soon as the debugging is done...
    # if sys.argv.__len__() == 2:
    #     cfg = get_cfg_defaults()
    #     cfg.merge_from_file(sys.argv[1])
    #     cfg.freeze()
    # else:
    #     raise Exception("You must specify a path to a yaml file to read the configuration from!")

    # Get the configuration
    cfg = get_cfg_defaults()
    cfg.merge_from_file("../cfg/train_kitti_monodepth.yaml")
    cfg.freeze()

    depth_trainer = train_depth.Trainer(cfg)
    depth_trainer.run()

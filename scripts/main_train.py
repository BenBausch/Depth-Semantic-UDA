# my_project/main.py
import sys
from train.depth import train_depth
from cfg.config import get_cfg_defaults  # local variable usage pattern, or:
# from config import cfg  # global singleton usage pattern


if __name__ == "__main__":
    if sys.argv.__len__() == 2:
        cfg = get_cfg_defaults()
        cfg.merge_from_file(sys.argv[1])
        cfg.freeze()
    else:
        raise Exception("You must specify a path to a yaml file to read the configuration from!")

    depth_trainer = train_depth.MonocularDepthFromMotionTrainer(cfg)
    depth_trainer.run()

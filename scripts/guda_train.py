# my_project/main.py
import sys
from train.guda import train_synthia_to_cityscapes
from cfg.config_training import create_configuration  # local variable usage pattern, or:
# from config import cfg  # global singleton usage pattern


if __name__ == "__main__":
    if sys.argv.__len__() == 2:
        cfg = create_configuration(sys.argv[1])
    else:
        raise Exception("You must specify a path to a yaml file to read the configuration from!")

    depth_trainer = train_synthia_to_cityscapes.GUDATrainer(cfg)
    depth_trainer.run()

# my_project/main.py
import sys

import torch.cuda
import torch.multiprocessing as mp
from train.base.train_base import run_trainer, setup_multi_processing
from train.depth_2_semantic import train_pointcloud_2_semantic
from cfg.config_training import create_configuration  # local variable usage pattern, or:
# from config import cfg  # global singleton usage pattern


if __name__ == "__main__":
    if sys.argv.__len__() == 2:
        cfg = create_configuration(sys.argv[1])
    else:
        raise Exception("You must specify a path to a yaml file to read the configuration from!")
    if cfg.device.multiple_gpus:
        # use DDP and 1 process per gpu for training
        setup_multi_processing()
        mp.spawn(run_trainer, nprocs=torch.cuda.device_count(),
                 args=(cfg, torch.cuda.device_count(), train_pointcloud_2_semantic.Depth2SemanticTrainer,))
    else:
        # standart single gpu 'cuda:0' training
        pc2s_trainer = train_pointcloud_2_semantic.Depth2SemanticTrainer(0, cfg)
        pc2s_trainer.run()



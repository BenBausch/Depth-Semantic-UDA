# default python packages
import sys

# own packages
import dataloaders
from cfg.config_dataset import get_cfg_dataset_defaults

# dependencies
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def evaluate_dataset(cfg):
    """Used to analyse depth ranges on a dataset"""
    dataset = dataloaders.get_dataset(cfg.dataset.name,
                                      'train',
                                      cfg.dataset.split,
                                      cfg)
    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False)

    for batch_idx, data in enumerate(loader):
        print(batch_idx)
        depth_mask = data['depth_dense'][0] > 100
        sky_mask = data['semantic'][0] == 10
        depth_mask = depth_mask.squeeze(0).numpy()
        num_pixels = depth_mask.shape[0] * depth_mask.shape[1]
        sky_mask = sky_mask.numpy()
        if np.sum(depth_mask != sky_mask) > num_pixels * 0.01:
            print(np.sum(depth_mask != sky_mask))
            plt.imshow(depth_mask != sky_mask)
            plt.show()
            print(batch_idx)
            plt.imshow(depth_mask)
            plt.show()
            plt.imshow(sky_mask)
            plt.show()


if __name__ == "__main__":
    path = sys.argv[1]
    print(f'Evaluating using {path} configuration file!')
    cfg = get_cfg_dataset_defaults()
    cfg.merge_from_file(path)
    cfg.freeze()
    evaluate_dataset(cfg)
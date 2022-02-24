# default python packages
import sys

# own packages
import dataloaders
from cfg.config_dataset import get_cfg_dataset_defaults

# dependencies
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def evaluate_model(cfg):
    dataset = dataloaders.get_dataset(cfg.dataset.name,
                                      'train',
                                      cfg.dataset.split,
                                      cfg)
    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=True)

    valid_classes, valid_names = dataset.get_valid_ids_and_names()
    valid_classes.append(dataset.ignore_index)
    valid_names.append('ignore_index')
    print(f'Valid Classes: {valid_classes}')
    print(f'Valid Class names: {valid_names}')

    class_pixel_count = [0.0 for i in range(len(valid_classes))]
    for batch_idx, data in enumerate(loader):
        print(batch_idx)
        names_count = ''
        data = data['semantic'].view(-1)
        for idx, cls_id in enumerate(valid_classes):
            names_count += valid_names[idx] + ': '
            class_pixel_count[idx] += torch.sum(data == cls_id).item() / data.shape[0]
            names_count += str(class_pixel_count[idx]) + ', '
        print(names_count[:-2])

    names_count = ''
    inv_names_count = ''
    for idx, cls_id in enumerate(valid_classes):
        names_count += valid_names[idx] + ': '
        inv_names_count += valid_names[idx] + ': '
        class_pixel_count[idx] = class_pixel_count[idx] / len(loader)
        names_count += str(class_pixel_count[idx]) + ', '
        inv_names_count += str(1/class_pixel_count[idx]) + ', '
    print('Percent of pixels belonging to each individual class across whole dataset')
    print(names_count[:-2])
    print(inv_names_count[:-2])
    print(class_pixel_count)
    print(1/class_pixel_count)


if __name__ == "__main__":
    path = sys.argv[1]
    print(f'Evaluating using {path} configuration file!')
    cfg = get_cfg_dataset_defaults()
    cfg.merge_from_file(path)
    cfg.freeze()
    evaluate_model(cfg)
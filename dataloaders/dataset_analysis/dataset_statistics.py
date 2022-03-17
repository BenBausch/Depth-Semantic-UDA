from torch.utils.data import DataLoader
import torch
from cfg.config_dataset import get_cfg_dataset_defaults
from dataloaders import get_dataset

if __name__== '__main__':
    path = r'C:\Users\benba\Documents\University\Masterarbeit\Depth-Semantic-UDA\cfg\yaml_files\train\kalimu_guda\kalimu.yaml'
    cfg = get_cfg_dataset_defaults()
    cfg.merge_from_file(path)
    cfg.freeze()

    torch.set_printoptions(precision=9)

    bs = 50

    kalimu = get_dataset('kalimu', 'train', None, cfg)
    loader = DataLoader(kalimu,
                        batch_size=bs,
                        shuffle=False,
                        num_workers=10,
                        pin_memory=True,
                        drop_last=False)

    mean = torch.tensor([0.0, 0.0, 0.0]).to('cuda:0')
    for idx, data in enumerate(loader):
        print(f'Processing {str(idx)} from {str(len(loader))}')
        data = data[('rgb', 0)].to('cuda:0')
        data = data.view(data.size(0), data.size(1), -1)
        mean += data.mean(2).sum(0) / bs
        print(f'Mean: {data.mean(2).sum(0) / bs}')
        print(f'Min {torch.min(data)}')
        print(f'Max {torch.max(data)}')
    print(mean / len(loader))
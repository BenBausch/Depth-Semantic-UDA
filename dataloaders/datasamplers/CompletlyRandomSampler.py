import torch
from torch.utils.data import Sampler, DataLoader


class CompletelyRandomSampler(Sampler):
    """
    Copied from https://discuss.pytorch.org/t/new-subset-every-epoch/85018
    and slightly modified by Ben Bausch
    """

    def __init__(self, data_source, num_samples=None):
        self.data_source = data_source
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        return iter(torch.randint(low=0, high=n, size=(self.num_samples,), dtype=torch.int64))

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    from dataloaders.cityscapes.dataset_cityscapes_semantic import CityscapesSemanticDataset
    from cfg.config_dataset import get_cfg_dataset_defaults
    from torch.utils.data import Sampler
    import matplotlib.pyplot as plt
    cfg = get_cfg_dataset_defaults()
    cfg.merge_from_file(
        r'C:\Users\benba\Documents\University\Masterarbeit\Depth-Semantic-UDA\cfg\yaml_files\train\guda\cityscapes_semantic.yaml')
    cfg.freeze()
    CITY_dataset = CityscapesSemanticDataset("train", None, cfg)

    img_h = cfg.dataset.feed_img_size[1]
    img_w = cfg.dataset.feed_img_size[0]
    batch_size = 1
    ds = DataLoader(CITY_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True,
                    sampler=CompletelyRandomSampler(data_source=CITY_dataset, num_samples=100000))

    for epoch in range(3):
        for i, data in enumerate(ds):
            print(data[('rgb', 0)].squeeze(0).numpy().shape)
            plt.imshow(data[('rgb', 0)].squeeze(0).numpy().transpose(1, 2, 0))
            plt.show()
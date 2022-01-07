import unittest
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from dataloaders.cityscapes.dataset_cityscapes_sequence import CityscapesSequenceDataset


class TestPathProcessing(unittest.TestCase):
    pass

class TestCityscapesSequenceDataset(unittest.TestCase):
    pass
    #def
    #DataLoader(train_dataset, batch_size=self.cfg.train.batch_size, shuffle=False,
    #           num_workers=self.cfg.train.nof_workers, pin_memory=True, drop_last=True)

if "__main__" == __name__:
    unittest.main()
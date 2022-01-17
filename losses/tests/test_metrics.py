import unittest

import numpy as np
import torch

from losses.metrics import MIoU


class TestMiou(unittest.TestCase):

    def test_miou_call(self):
        mask_gt = torch.tensor([[[[0], [1], [2]],
                                 [[0], [0], [0]]]]).transpose(1, 2).transpose(1, 3)

        mask_pred = torch.tensor([[[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                                   [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]]).transpose(1, 2).transpose(1,
                                                                                                                    3)

        m_iou = MIoU(num_classes=3)

        m_iou.update(mask_pred=mask_pred, mask_gt=mask_gt)

        miou, iou_per_class = m_iou.get_miou()

        miou_gt = 0.2

        iou_per_class_gt = torch.tensor([0.6, 0.0, 0.0])

        # round values
        precision = 4
        miou = np.true_divide(np.floor(miou * 10 ** precision), 10 ** precision)
        for i, val in enumerate(iou_per_class):
            iou_per_class[i] = np.true_divide(np.floor(val * 10 ** precision), 10 ** precision)

        self.assertTrue(miou == miou_gt)
        self.assertTrue(torch.all(iou_per_class == iou_per_class_gt))


if "__main__" == __name__:
    unittest.main()



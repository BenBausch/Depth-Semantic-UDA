# Own packages
from losses.losses import *
from camera_models import PinholeCameraModel

# Python packages
import torch
import torch.nn as nn
import unittest
import numpy as np
import math


class TestSurfaceNormalRegularizationLoss(unittest.TestCase):  # todo: update test

    def test_cosine_similarity(self):
        use_less_camera = PinholeCameraModel(2, 2, 1, 1, 1, 1)  # just needed for initializing the loss, not used in
        # this test
        loss = SurfaceNormalRegularizationLoss(use_less_camera, 'cpu')

        normals_gt = torch.tensor(
            [[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
              [[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]]]]
        ).transpose(1, 3)

        normals_pred = torch.tensor(
            [[[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]],
              [[4.0, 0.0, 3.0], [5.0, 0.0, 2.0]]]]
        ).transpose(1, 3)

        self.assertTrue(torch.equal(loss.cosine_similarity_guda(normals_pred, normals_gt), torch.tensor([0.325])))

    def test_get_normal_vectors(self):
        use_less_camera = PinholeCameraModel(4, 3, 1, 1, 1, 1)  # just needed for initializing the loss, not used in
        # this test
        loss = SurfaceNormalRegularizationLoss(use_less_camera, 'cpu')
        points3d = torch.tensor([[[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                                  [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                                  [[0.0, 0.0, 2.0], [0.0, 2.0, 1.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0]]]]
                                ).transpose(1, 3)

        normals_gt = torch.tensor(
            [[[[0.0, 0.0, 0.0], [-0.707107, -0.707107, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
              [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-0.707107, 0.0, 0.707107], [1.0, 0.0, 0.0]],
              [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -0.447214, -0.894427],
               [0, 0.707107, 0.707107]]]]
        ).transpose(1, 3)

        normals = loss.get_normal_vectors(points3d)

        precision = 4
        rounded_gt = np.true_divide(np.floor(normals_gt * 10 ** precision), 10 ** precision)
        rounded_normals = np.true_divide(np.floor(normals * 10 ** precision), 10 ** precision)

        self.assertTrue(torch.equal(rounded_gt, rounded_normals))


class TestBootstrappedCrossEntropy(unittest.TestCase):

    def test_decaying_ratio(self):
        bce = BootstrappedCrossEntropy(img_height=10, img_width=10, r=0.3, start_decay_epoch=2, end_decay_epoch=7)
        ratios = [1.0, 1.0, 1.0, 0.86, 0.72, 0.58, 0.44, 0.30, 0.30, 0.30]
        ks = [100, 100, 100, 86, 72, 58, 44, 30, 30, 30]

        pred = torch.DoubleTensor([[[[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
                                    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
                                    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
                                    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
                                    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
                                    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
                                    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
                                    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
                                    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
                                    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
                                     [1, 0]]]]).transpose(1, 3)

        gt = torch.LongTensor([[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]])

        for epoch in range(10):
            bce(pred, gt, epoch)
            assert round(bce.ratio, 2) == ratios[epoch]
            assert round(bce.k, 2) == ks[epoch]

    def test_normal_case(self):
        prediction = torch.tensor([[[0.3, 0.7, 0.0], [0.8, 0.2, 0.0]],
                                   [[0.1, 0.9, 0.0], [0.6, 0.4, 0.0]]],
                                  requires_grad=True).transpose(0, 2).transpose(1, 2).unsqueeze(0)

        target = torch.tensor([[0, 0],
                               [1, 2]], dtype=torch.long).unsqueeze(0)

        ratio = 0.5  # worst 2 prediction is pixel [0,0] with 0.3 and [1,1] with 0.0
        btce = BootstrappedCrossEntropy(img_height=2, img_width=2, r=ratio)

        loss_flat = [1.203972816, 1.203972816, 0.105360538,
                     20.723266602]  # calculated using well implement cross_entropy loss from pytorch
        worst_2_avg = (loss_flat[0] + loss_flat[3]) / 2

        worst_2_avg = round(worst_2_avg, 4)
        loss_btce_value = round(btce(prediction, target).item(), 4)

        assert worst_2_avg == loss_btce_value

    def test_ignore_index(self):
        prediction = torch.tensor([[[0.3, 0.7, 0.0], [0.8, 0.2, 0.0]],
                                   [[0.1, 0.9, 0.0], [0.6, 0.4, 0.0]]],
                                  requires_grad=True).transpose(0, 2).transpose(1, 2).unsqueeze(0)

        target = torch.tensor([[0, 0],
                               [1, 250]], dtype=torch.long).unsqueeze(0)

        ratio = 0.5  # worst 2 prediction is pixel [0,0] with 0.3 and [0, 1] with 0.8, since pixel [1,1] will be ignored
        btce = BootstrappedCrossEntropy(img_height=2, img_width=2, r=ratio, ignore_index=250)

        loss_flat = [1.203972816, 0.223143533, 0.105360538]
        worst_2_avg = (loss_flat[0] + loss_flat[1]) / 2

        worst_2_avg = math.floor(worst_2_avg * 100000) / 100000
        loss_btce_value = math.floor(btce(prediction, target).item() * 100000) / 100000  # floor to 5 decimal points

        assert worst_2_avg == loss_btce_value

    def test_multiple_same_losses(self):
        prediction = torch.tensor([[[0.3, 0.7, 0.0], [0.3, 0.7, 0.0]],
                                   [[0.1, 0.9, 0.0], [0.3, 0.7, 0.0]]],
                                  requires_grad=True).transpose(0, 2).transpose(1, 2).unsqueeze(0)

        target = torch.tensor([[0, 0],
                               [1, 0]], dtype=torch.long).unsqueeze(0)

        ratio = 0.5  # worst 2 prediction is pixel [0,0] with 0.3 and [0, 1] with 0.8, since pixel [1,1] will be ignored
        btce = BootstrappedCrossEntropy(img_height=2, img_width=2, r=ratio, ignore_index=250)

        loss_flat = [1.203972816, 1.203972816, 0.105360538, 1.203972816]
        worst_2_avg = (loss_flat[0] + loss_flat[1]) / 2

        worst_2_avg = math.floor(worst_2_avg * 10000) / 10000
        loss_btce_value = math.floor(btce(prediction, target).item() * 10000) / 10000  # floor to 4 decimal points

        assert worst_2_avg == loss_btce_value


class TestL1Loss(unittest.TestCase):
    def setUp(self):
        # Create two dummy tensors on the cpu
        self.diff_tensor = torch.tensor([[1, 2, 4, 5], [1, 2, 4, 5]]).float()

        depth = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
        self.mask = depth > 0

    def test_masking(self):

        mean_unmasked = self.diff_tensor.mean()
        mean_masked = self.diff_tensor[self.mask].mean()

        self.assertAlmostEqual(mean_unmasked, 3.0)
        self.assertAlmostEqual(mean_masked, 4.5)


if "__main__" == __name__:
    unittest.main()

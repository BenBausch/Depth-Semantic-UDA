import torch
import unittest

class Test_L1Loss(unittest.TestCase):
    def setUp(self):
        # Create two dummy tensors on the cpu
        self.diff_tensor = torch.tensor([[1, 2, 4, 5], [1, 2, 4, 5]]).float()

        depth = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
        self.mask = depth > 0

    def test_masking(self):
        print("Unmasked:", self.diff_tensor)
        print("Mask:", self.mask)
        print("Masked:", self.diff_tensor[self.mask])

        mean_unmasked = self.diff_tensor.mean()
        mean_masked = self.diff_tensor[self.mask].mean()

        self.assertAlmostEqual(mean_unmasked, 3.0)
        self.assertAlmostEqual(mean_masked, 4.5)

        print("Expected unmasked mean:", 3.0)
        print("Actual unmasked mean:", mean_unmasked)

        print("Expected masked mean:", 4.5)
        print("Actual masked mean:", mean_masked)

if "__main__" == __name__:
    unittest.main()
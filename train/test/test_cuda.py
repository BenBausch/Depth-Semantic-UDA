# This is a simple test to demonstrate that an error is thrown if tensors are processed together that are not
# on the same device
import torch
import torch.nn
import unittest

class Test_CrossComputation(unittest.TestCase):
    def setUp(self):
        # Create two dummy tensors on the cpu
        self.a_cpu = torch.tensor([2, 0, 3])
        self.b_cpu = torch.tensor([1, 4, 2])

        # Duplicate both sensors and move to gpu
        self.a_gpu = self.a_cpu.cuda()
        self.b_gpu = self.b_cpu.cuda()

    def test_cpu_computation(self):
        # Multiply two cpu tensors (should work fine)
        try:
            c_cpu = self.a_cpu * self.b_cpu
            print("c_cpu", c_cpu)
            assert True
        except:
            assert False

    def test_gpu_computation(self):
        # Multiply two gpu tensors (works fine)
        try:
            c_gpu = self.a_gpu * self.b_gpu
            print("c_gpu", c_gpu)
            assert True
        except:
            assert False

    def test_cross_computation(self):
        # Cross multiplication (should throw an error)
        try:
            c_cross = self.a_gpu * self.b_cpu
            print("c_cross", c_cross)
            assert False
        except:
            assert True

if "__main__" == __name__:
    unittest.main()
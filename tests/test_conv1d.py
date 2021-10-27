from conv1d import conv1d
import unittest
import torch
import torch.nn.functional as F


class MyConv1dTestCase(unittest.TestCase):
    def test_conv1d_1(self):
        input = torch.tensor([[[3, 2, 1, 5, 6, 7]], [[2, 3, 4, 0, -1, 0]]], dtype=torch.float)
        weight = torch.tensor([[[1, 2, 3]]], dtype=torch.float)
        bias = torch.tensor([4], dtype=torch.float)
        expected_result = [[[14, 23, 33, 42]], [[24, 15, 5, 2]]]

        their_result = F.conv1d(input, weight, bias)
        my_result = conv1d(input, weight, bias)

        self.assertListEqual(their_result.tolist(), expected_result)
        self.assertListEqual(my_result.tolist(), expected_result)

    def test_conv1d_2(self):
        input = torch.tensor([[[3, 2, 1, 5, 6, 7], [0, 1, 0, -1, 0, 0]],
                              [[2, 3, 4, 0, -1, 0], [1, 1, 1, -1, -1, -1]]], dtype=torch.float)
        weight = torch.tensor([[[1, 2, 3], [1, 0, -1]]], dtype=torch.float)
        bias = torch.tensor([4], dtype=torch.float)
        expected_result = [[[14, 25, 33, 41]], [[24, 17, 7, 2]]]

        their_result = F.conv1d(input, weight, bias)
        my_result = conv1d(input, weight, bias)

        self.assertListEqual(their_result.tolist(), expected_result)
        self.assertListEqual(my_result.tolist(), expected_result)

    def test_conv1d_3(self):
        input = torch.tensor([[[3, 2, 1, 5, 6, 7], [0, 1, 0, -1, 0, 0]],
                              [[2, 3, 4, 0, -1, 0], [1, 1, 1, -1, -1, -1]]], dtype=torch.float)
        weight = torch.tensor([[[1, 2, 3], [1, 0, -1]], [[0, 0, 1], [1, 0, 0]]], dtype=torch.float)
        bias = torch.tensor([4, 3], dtype=torch.float)
        expected_result = [[[14, 25, 33, 41], [4, 9, 9, 9]], [[24, 17, 7, 2], [8, 4, 3, 2]]]

        their_result = F.conv1d(input, weight, bias)
        my_result = conv1d(input, weight, bias)

        self.assertListEqual(their_result.tolist(), expected_result)
        self.assertListEqual(my_result.tolist(), expected_result)

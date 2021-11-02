from convNd import convNd
from collections import namedtuple
import unittest
import torch
import torch.nn.functional as F
from parameterized import parameterized

SampleData = namedtuple("SampleData", ["input", "weight", "bias", "stride", "padding",
                                       "expected_result"])

examples1d = [
    SampleData(
        input=torch.tensor([[[3, 2, 1, 5, 6, 7]], [[2, 3, 4, 0, -1, 0]]], dtype=torch.float),
        weight=torch.tensor([[[1, 2, 3]]], dtype=torch.float),
        bias=torch.tensor([4], dtype=torch.float),
        stride=1,
        padding=0,
        expected_result=[[[14, 23, 33, 42]], [[24, 15, 5, 2]]]
    ),
    SampleData(
        input=torch.tensor([[[3, 2, 1, 5, 6, 7]], [[2, 3, 4, 0, -1, 0]]], dtype=torch.float),
        weight=torch.tensor([[[1, 2, 3]]], dtype=torch.float),
        bias=torch.tensor([4], dtype=torch.float),
        stride=2,
        padding=0,
        expected_result=[[[14, 33]], [[24, 5]]]
    ),
    SampleData(
        input=torch.tensor([[[3, 2, 1, 5, 6, 7]], [[2, 3, 4, 0, -1, 0]]], dtype=torch.float),
        weight=torch.tensor([[[1, 2, 3]]], dtype=torch.float),
        bias=torch.tensor([4], dtype=torch.float),
        stride=2,
        padding=1,
        expected_result=[[[16, 23, 42]], [[17, 15, 2]]]
    ),
    SampleData(
        input=torch.tensor([[[3, 2, 1, 5, 6, 7]], [[2, 3, 4, 0, -1, 0]]], dtype=torch.float),
        weight=torch.tensor([[[1, 2, 3]]], dtype=torch.float),
        bias=None,
        stride=1,
        padding=0,
        expected_result=[[[10, 19, 29, 38]], [[20, 11, 1, -2]]]
    ),
    SampleData(
        input=torch.tensor([[[3, 2, 1, 5, 6, 7], [0, 1, 0, -1, 0, 0]],
                            [[2, 3, 4, 0, -1, 0], [1, 1, 1, -1, -1, -1]]],
                           dtype=torch.float),
        weight=torch.tensor([[[1, 2, 3], [1, 0, -1]]], dtype=torch.float),
        bias=torch.tensor([4], dtype=torch.float),
        stride=1,
        padding=0,
        expected_result=[[[14, 25, 33, 41]], [[24, 17, 7, 2]]]
    ),
    SampleData(
        input=torch.tensor([[[3, 2, 1, 5, 6, 7], [0, 1, 0, -1, 0, 0]],
                            [[2, 3, 4, 0, -1, 0], [1, 1, 1, -1, -1, -1]]],
                           dtype=torch.float),
        weight=torch.tensor([[[1, 2, 3], [1, 0, -1]]], dtype=torch.float),
        bias=torch.tensor([4], dtype=torch.float),
        stride=2,
        padding=0,
        expected_result=[[[14, 33]], [[24, 7]]]
    ),
    SampleData(
        input=torch.tensor([[[3, 2, 1, 5, 6, 7], [0, 1, 0, -1, 0, 0]],
                            [[2, 3, 4, 0, -1, 0], [1, 1, 1, -1, -1, -1]]],
                           dtype=torch.float),
        weight=torch.tensor([[[1, 2, 3], [1, 0, -1]]], dtype=torch.float),
        bias=torch.tensor([4], dtype=torch.float),
        stride=2,
        padding=2,
        expected_result=[[[13, 14, 33, 24]], [[9, 24, 7, 2]]]
    ),
    SampleData(
        input=torch.tensor([[[3, 2, 1, 5, 6, 7], [0, 1, 0, -1, 0, 0]],
                            [[2, 3, 4, 0, -1, 0], [1, 1, 1, -1, -1, -1]]],
                           dtype=torch.float),
        weight=torch.tensor([[[1, 2, 3], [1, 0, -1]]], dtype=torch.float),
        bias=None,
        stride=1,
        padding=0,
        expected_result=[[[10, 21, 29, 37]], [[20, 13, 3, -2]]]
    ),
    SampleData(
        input=torch.tensor([[[3, 2, 1, 5, 6, 7], [0, 1, 0, -1, 0, 0]],
                            [[2, 3, 4, 0, -1, 0], [1, 1, 1, -1, -1, -1]]],
                           dtype=torch.float),
        weight=torch.tensor([[[1, 2, 3], [1, 0, -1]], [[0, 0, 1], [1, 0, 0]]], dtype=torch.float),
        bias=torch.tensor([4, 3], dtype=torch.float),
        stride=1,
        padding=0,
        expected_result=[[[14, 25, 33, 41], [4, 9, 9, 9]], [[24, 17, 7, 2], [8, 4, 3, 2]]]
    ),
    SampleData(
        input=torch.tensor([[[3, 2, 1, 5, 6, 7], [0, 1, 0, -1, 0, 0]],
                            [[2, 3, 4, 0, -1, 0], [1, 1, 1, -1, -1, -1]]],
                           dtype=torch.float),
        weight=torch.tensor([[[1, 2, 3], [1, 0, -1]], [[0, 0, 1], [1, 0, 0]]], dtype=torch.float),
        bias=torch.tensor([4, 3], dtype=torch.float),
        stride=2,
        padding=0,
        expected_result=[[[14, 33], [4, 9]], [[24, 7], [8, 3]]]
    ),
    SampleData(
        input=torch.tensor([[[3, 2, 1, 5, 6, 7], [0, 1, 0, -1, 0, 0]],
                            [[2, 3, 4, 0, -1, 0], [1, 1, 1, -1, -1, -1]]],
                           dtype=torch.float),
        weight=torch.tensor([[[1, 2, 3], [1, 0, -1]], [[0, 0, 1], [1, 0, 0]]], dtype=torch.float),
        bias=None,
        stride=1,
        padding=0,
        expected_result=[[[10, 21, 29, 37], [1, 6, 6, 6]], [[20, 13, 3, -2], [5, 1, 0, -1]]]
    )
]


examples2d = [
    SampleData(
        input=torch.tensor([[[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11],
                              [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23]]]],
                           dtype=torch.float),
        weight=torch.tensor([[[[1, 0, 1], [0, -1, 1]]]], dtype=torch.float),
        bias=torch.tensor([4], dtype=torch.float),
        stride=1,
        padding=0,
        expected_result=[[[[7, 9, 11, 13], [19, 21, 23, 25], [31, 33, 35, 37]]]]
    ),
    SampleData(
        input=torch.tensor([[[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11],
                              [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23]]]],
                           dtype=torch.float),
        weight=torch.tensor([[[[1, 0, 1], [0, -1, 1]]]], dtype=torch.float),
        bias=torch.tensor([4], dtype=torch.float),
        stride=3,
        padding=2,
        expected_result=[[[[4, 4, 4], [22, 21, -3], [4, 4, 4]]]]
    ),
    SampleData(
        input=torch.tensor([[[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11],
                              [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23]]]],
                           dtype=torch.float),
        weight=torch.tensor([[[[1, 0, 1], [0, -1, 1]]]], dtype=torch.float),
        bias=None,
        stride=1,
        padding=0,
        expected_result=[[[[3, 5, 7, 9], [15, 17, 19, 21], [27, 29, 31, 33]]]]
    ),
]


class MyConvNdTestCase(unittest.TestCase):
    @parameterized.expand([example._asdict().values() for example in examples1d])
    def test_conv1d(self, input, weight, bias, stride, padding, expected_result):
        their_result = F.conv1d(input, weight, bias, stride, padding)
        my_result = convNd(input, weight, bias, stride, padding)

        self.assertListEqual(their_result.tolist(), expected_result)
        self.assertListEqual(my_result.tolist(), expected_result)

    @parameterized.expand([example._asdict().values() for example in examples2d])
    def test_conv2d(self, input, weight, bias, stride, padding, expected_result):
        their_result = F.conv2d(input, weight, bias, stride, padding)
        my_result = convNd(input, weight, bias, stride, padding)

        self.assertListEqual(their_result.tolist(), expected_result)
        self.assertListEqual(my_result.tolist(), expected_result)

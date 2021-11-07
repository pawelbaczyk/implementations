import unittest
from hmc import HMC
import numpy


class HMCTestCase(unittest.TestCase):
    def test_initialization_of_hmc(self):
        def func(theta):
            return numpy.exp(-(numpy.dot(theta, theta))) / (2 * numpy.pi)

        theta_zero = numpy.array([0, 0])

        sampler = HMC(probability=func, theta_zero=theta_zero)

    def test_sample(self):
        def func(theta):
            return numpy.exp(-1 / 2 * (numpy.dot(theta, theta))) / (2 * numpy.pi)

        theta_zero = numpy.array([0, 0])
        N = 100

        sampler = HMC(probability=func, theta_zero=theta_zero)
        samples = sampler.sample(number_of_samples=N, epsilon=0.1, L=3)
        self.assertEqual(len(samples), N)

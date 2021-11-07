import unittest
from hmc import HMC
import numpy
import matplotlib.pyplot as plt


class HMCTestCase(unittest.TestCase):
    def test_initialization_of_hmc(self):
        def func(theta):
            return numpy.log(numpy.exp(-(numpy.dot(theta, theta))) / (2 * numpy.pi))

        def grad(theta):
            return 1 / func(theta) * -2 * func(theta) * theta

        theta_zero = numpy.array([0, 0])

        sampler = HMC(p=func, grad=grad, theta_zero=theta_zero)

    def test_sample(self):
        def func(theta):
            return numpy.log(numpy.exp(-1 / 2 * (numpy.dot(theta, theta))) / (2 * numpy.pi))

        def grad(theta):
            return 1 / func(theta) * -2 * func(theta) * theta

        theta_zero = numpy.array([0, 0])
        N = 100

        sampler = HMC(p=func, grad=grad, theta_zero=theta_zero)
        samples = sampler.sample(number_of_samples=N, epsilon=0.1, L=3)
        self.assertEqual(len(samples), N)


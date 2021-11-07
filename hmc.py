import numpy
import numpy as np


class HMC:
    def __init__(self, p, grad, theta_zero):
        self.p = p
        self.grad = grad
        self.theta_m_minus_one = theta_zero

    def sample(self, number_of_samples, epsilon, L):
        samples = []
        while len(samples) < number_of_samples:
            single_sample = self._get_single_sample(epsilon, L)
            if single_sample is not None:
                samples.append(single_sample)
        return numpy.array(samples)

    def _get_single_sample(self, epsilon, L):
        theta_tilde = theta_m = self.theta_m_minus_one
        r_tilde = r_zero = self._sample_momentum(theta_m)

        for _ in range(L):
            theta_tilde, r_tilde = self._leapfrog(theta_tilde, r_tilde, epsilon)

        if self._is_accepted_by_metropolis(theta_tilde, r_tilde, r_zero):
            self.theta_m_minus_one = theta_tilde
            return theta_tilde

    def _is_accepted_by_metropolis(self, theta_tilde, r_tilde, r_zero):
        alpha = np.minimum(1, self._exponent(theta_tilde, r_tilde) / self._exponent(self.theta_m_minus_one, r_zero))
        return numpy.random.uniform() < alpha

    def _exponent(self, theta, r):
        return numpy.exp(self.p(theta) - 1 / 2 * numpy.dot(r, r))

    @staticmethod
    def _sample_momentum(theta):
        return numpy.random.multivariate_normal(mean=numpy.zeros_like(theta), cov=numpy.eye(N=len(theta)))

    def _leapfrog(self, theta, r, epsilon):
        r_tilde = r + epsilon / 2 * self.grad(theta)
        theta_tilde = theta + epsilon * r_tilde
        r_tilde += epsilon / 2 * self.grad(theta_tilde)
        return theta_tilde, r_tilde

import numpy
import numpy as np
import numdifftools


class HMC:
    def __init__(self, probability, theta_zero):
        self._probability = probability
        self._theta_m_minus_one = theta_zero

    def _log_p(self, args, *kwargs):
        return numpy.log(self._probability(args, *kwargs))

    def _grad_of_log_p(self, args, *kwargs):
        return numdifftools.Gradient(self._log_p)(args, *kwargs)

    def sample(self, number_of_samples, epsilon, L):
        samples = []
        while len(samples) < number_of_samples:
            single_sample = self._get_single_sample(epsilon, L)
            if single_sample is not None:
                samples.append(single_sample)
        return numpy.array(samples)

    def _get_single_sample(self, epsilon, L):
        theta_tilde = theta_m = self._theta_m_minus_one
        r_tilde = r_zero = self._sample_momentum(theta_m)

        for _ in range(L):
            theta_tilde, r_tilde = self._leapfrog(theta_tilde, r_tilde, epsilon)

        if self._is_accepted_by_metropolis(theta_tilde, r_tilde, r_zero):
            self._theta_m_minus_one = theta_tilde
            return theta_tilde

    def _is_accepted_by_metropolis(self, theta_tilde, r_tilde, r_zero):
        alpha = np.minimum(1, self._exponent(theta_tilde, r_tilde) / self._exponent(self._theta_m_minus_one, r_zero))
        return numpy.random.uniform() < alpha

    def _exponent(self, theta, r):
        return numpy.exp(self._log_p(theta) - 1 / 2 * numpy.dot(r, r))

    @staticmethod
    def _sample_momentum(theta):
        return numpy.random.multivariate_normal(mean=numpy.zeros_like(theta), cov=numpy.eye(N=len(theta)))

    def _leapfrog(self, theta, r, epsilon):
        r_tilde = r + epsilon / 2 * self._grad_of_log_p(theta)
        theta_tilde = theta + epsilon * r_tilde
        r_tilde += epsilon / 2 * self._grad_of_log_p(theta_tilde)
        return theta_tilde, r_tilde

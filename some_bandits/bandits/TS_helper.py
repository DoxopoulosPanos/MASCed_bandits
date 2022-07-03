import numpy as np

class Arm:
    def __init__(self, arm):
        self.arm = arm
        self.successes = 0
        self.failures = 0
        self.prior = "beta"
        self.posterior = "bernoulli"
        self.probability = 0

    def sample(self):
        self.probability = self.sample_beta_prior()[0]

    def sample_beta_prior(self, total_samples=1):
        return np.random.beta(self.successes + 1, self.failures + 1, total_samples)
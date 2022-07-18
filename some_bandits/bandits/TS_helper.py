import numpy as np

class Arm:
    def __init__(self, arm):
        self.arm = arm
        self.successes = 0
        self.failures = 0
        self.prior = "beta"
        self.posterior = "bernoulli"
        self.probability = 0

        self.mu = 0
        self.sigma = 10000
        self.rewards = []

    def sample(self, formula):
        if formula == "BB":
            self.probability = self.sample_beta_prior()[0]
        elif formula == "NN":
            self.probability = self.sample_normal_distribution()[0]
        else:
            raise RuntimeError("No formula specified")

    def sample_beta_prior(self, total_samples=1):
        return np.random.beta(self.successes + 1, self.failures + 1, total_samples)

    def sample_normal_distribution(self, total_samples=1):
        return np.random.normal(self.mu, self.sigma, total_samples)
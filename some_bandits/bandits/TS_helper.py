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
        self.real_sigma = 1000
        self.rewards = []

    def sample(self, formula):
        if formula == "BB":
            self.probability = self.sample_beta_prior()[0]      ### TODO: this should be called sample not probability
        elif formula == "NN":
            self.probability = self.sample_normal_distribution()[0]
        else:
            raise RuntimeError("No formula specified")

    ### Beta Posterior
    def sample_beta_prior(self, total_samples=1):
        return np.random.beta(self.successes + 1, self.failures + 1, total_samples)

    ### Normal Posterior
    def sample_normal_distribution(self, total_samples=1):
        return np.random.normal(self.mu, self.sigma**2 + self.real_sigma**2, total_samples)

    def calculate_new_mu_and_sigma(self):
        times_played = len(self.rewards)
        total_reward = sum(self.rewards)
        new_sigma = ((1/self.sigma)**2 + (times_played / (self.real_sigma**2)))**(-1)
        temp = self.mu/(self.sigma**2) + total_reward/(self.real_sigma**2)
        self.mu = temp * new_sigma
        self.sigma = new_sigma



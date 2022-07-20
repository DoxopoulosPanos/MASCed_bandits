import numpy as np

class Arm:
    def __init__(self, arm, mu=None, sigma=None):
        if (mu is None) and (sigma is None):
            self.arm = arm
            self.successes = 0
            self.failures = 0
            self.prior = "beta"
            self.posterior = "bernoulli"
            self.probability = 0

            self.mu_0 = 0.5
            self.sigma_0 = 0.7
            self.sigma = 0.1
            self.rewards = []
        else:
            """
            used for testing purposes
            """
            self.arm = arm
            self.mu = mu
            self.sigma = sigma


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
    # def sample_normal_distribution(self, total_samples=1):
    #     return np.random.normal(self.mu, self.variance + self.real_variance, total_samples)
    def sample_normal_distribution(self, total_samples=1):
        sample = np.random.normal(self.mu_0, self.sigma_0, total_samples)
        # print("Arm: {}".format(self.arm))
        # print("mu_0: {}".format(self.mu_0))
        # print("sigma_0: {}".format(self.sigma_0))
        # print("sample: {}".format(sample))
        return sample

    def calculate_new_mu_and_sigma(self):
        times_played = len(self.rewards)
        total_reward = sum(self.rewards)
        print("sigma: {}".format(self.sigma_0))
        new_sigma_squared = ((1/self.sigma_0**2) + (times_played / (self.sigma**2)))**(-1)
        temp = self.mu_0/(self.sigma_0**2) + total_reward/(self.sigma**2)
        self.mu_0 = temp * new_sigma_squared
        self.sigma_0 = np.sqrt(new_sigma_squared)


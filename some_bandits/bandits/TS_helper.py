import numpy as np

class Arm:
    def __init__(self, arm, mu=None, sigma=None, a=None, b=None):
        if (mu is None) and (sigma is None) and (a is None) and (b is None):
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
        elif (a is None) and (b is None):
            """
            used for testing purposes
            """
            self.arm = arm
            self.mu = mu
            self.sigma = sigma
        elif (a is not None) and (b is not None):
            # Normal with normal-gamma prior
            self.init_normal_with_gamma_posterior()


    def init_normal_with_gamma_posterior(self):
        self.a = 1  # shape
        self.b = 10  # rate
        self.mu_0 = 1 # the prior (estimated) mean
        self.nu  = 0    # total observations
        self.nu_0 = self.b / (self.a + 1) # the prior (estimated) variance 
        self.rewards = []


    def sample(self, formula):
        if formula == "BB":
            self.probability = self.sample_beta_posterior()[0]      ### TODO: this should be called sample not probability
        elif formula == "NN":
            self.probability = self.sample_normal_distribution()[0]
        elif formula == "NGN":
            self.probability = self.sample_normal_with_gamma_prior()[0] 
        else:
            raise RuntimeError("No formula specified")

    ### Beta Posterior
    def sample_beta_posterior(self, total_samples=1):
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

    def sample_normal_with_gamma_prior(self, total_samples=1):           ### TODO: check this
        precision = np.random.gamma(self.a, 1/self.b)
        if precision == 0 or self.n == 0: 
            precision = 0.001
        
        variance = 1/precision
        return np.random.normal( self.mu_0, np.sqrt(variance), total_samples)

    def calculate_new_mu_and_sigma(self):
        observations = len(self.rewards)
        total_reward = sum(self.rewards)
        print("sigma: {}".format(self.sigma_0))
        new_sigma_squared = ((1/self.sigma_0**2) + (observations / (self.sigma**2)))**(-1)
        temp = self.mu_0/(self.sigma_0**2) + total_reward/(self.sigma**2)
        self.mu_0 = temp * new_sigma_squared
        self.sigma_0 = np.sqrt(new_sigma_squared)

    def update_normal_posterior_with_normal_gamma_prior(self):
        
        # we have 1 sample reward so n=1 and mean_reward_samples=reward
        n = 1
        self.a = self.a + n/2
        self.b = self.b +  n * self.nu / (n + self.nu) * 1/2 * (self.rewards[-1] - self.mu_0)**2  # the SUM term is zero, since there is only one sample
        self.mu_0 = mean(self.rewards)
        self.nu = self.nu + n # increase the number of observations by 1

        
        ###### WIKI algorithm self.rewards should be the last ONLY sample (reward)
        # # the number of sample values seen since the last time step
        # n = 1
        # mean_reward = mean(self.rewards)

        # self.a = self.a + n/2
        # self.b = self.b + 1/2 * sum((np.array(self.rewards) - mean_reward)**2) + (n * self.nu)/(self.nu + n) * ((mean_reward - self.mu_0)**2)/2
        # self.mu_0 = ((self.nu * self.mu_0) + (n * mean_reward)) / (self.nu + n)
        # self.nu = self.nu + n  # increase total observations

        # self.nu_0 = self.b / (self.a + 1) # new variance
        



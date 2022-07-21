#!/usr/bin/env python

import numpy as np
from some_bandits.bandits.TS_helper import Arm
from some_bandits.bandits.TS import TS
from scipy.stats import ks_2samp

class Test():
    def __init__(self):
        self.test = "NN"
        self.config_arms = []
        self.desired_arm = None

    def init_3_arms(self):
        # desired_distribution
        arm_value = None
        mu = 0.6
        sigma = 0.2
        self.desired_arm = Arm(arm_value, mu, sigma)
        ###########

        arms = []
        # Arm (3,1)
        arm_value = (3,1)
        mu = 0.2
        sigma = 0.5
        arm = Arm(arm_value, mu, sigma)
        self.config_arms.append(arm)

        # Arm (2,1)
        arm_value = (2,1)
        mu = 0.5
        sigma = 0.5
        arm = Arm(arm_value, mu, sigma)
        self.config_arms.append(arm)

        # Arm (1,1)
        arm_value = (1,1)
        mu = 0.6
        sigma = 0.2
        arm = Arm(arm_value, mu, sigma)
        self.config_arms.append(arm)

    def return_reward(self, arm):
        # compare real 
        return arm.sample_normal_distribution()

    def get_probability_from_kolmogorov_smirnov_test(self, mu1, sigma1, mu2, sigma2):
        np.random.seed(12345678)
        x = np.random.normal(mu1, sigma1, 50000)
        y = np.random.normal(mu2, sigma2, 50000)
        return(ks_2samp(x, y).pvalue)

    def get_reward(self, mu1, mu2):
        return 1 - abs(mu2 - mu1)


    def test_execution(self):
        all_actions = []
        ts = TS(self.test)

        for i in range(100):
            print(ts.last_action)
            all_actions.append(ts.last_action)

            for ts_arm in self.config_arms:
                if ts_arm.arm == ts.last_action:
                    # this is the played arm. Calculate the reward
                    # reward = self.get_probability_from_kolmogorov_smirnov_test(ts_arm.mu, ts_arm.sigma, self.desired_arm.mu, self.desired_arm.sigma)
                    reward = self.get_reward(ts_arm.mu, self.desired_arm.mu)
            ts.start_strategy(reward)
        
        print("\n")
        print("Actions performed: ")
        print(*all_actions, sep = ", ") 

if "__name__" == "__main__":
    from some_bandits.bandits.test_bandits.TS_test import Test
    test = Test()
    test.init_3_arms()
    test.test_execution()




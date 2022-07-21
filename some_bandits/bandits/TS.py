from operator import attrgetter
import numpy as np
import time
from random import choice, sample
#from utilities import save_to_pickle, load_from_pickle, truncate, convert_conf, calculate_utility
from some_bandits.bandit_options import bandit_args
from some_bandits.bandits.Bandit import Bandit
import matplotlib.pyplot as plt
from some_bandits.bandits.TS_helper import Arm as TS_arm

PRIOR_FYNC = None
POSTERIOR_FYNC = None

class TS(Bandit):
    def __init__(self, formula):
        super().__init__("TS-" + formula )
        self.ts_arms = []
        initial_configuration = bandit_args["initial_configuration"]
        self.game_list = []
        self.last_action = initial_configuration
        self.formula = formula

        # initialise arms and probabilities
        for arm in self.arms:
            self.ts_arms.append(TS_arm(arm))

    def start_strategy(self, reward):
        self.game_list.append([reward, self.last_action])

        if self.formula == "BB":  # Beta Prior, Binomial Posterior
            # update values based on reward
            self.update_posterior(reward)
        elif self.formula == "NN":   # Normal Prior Normal Posterior, known variance
            self.update_posterior_normal(reward)
        elif self.formula == "NGN":   # Normal Gamma Prior, Normal Posterior (unknown mean and variance)
            self.update_posterior_normal_from_gamma(reward)


        # iterate for each arm
        for ts_arm in self.ts_arms:
            ts_arm.sample(self.formula)

        for ts_arm in self.ts_arms:
            print("this is the arm: {} and this is the probability: {}".format(ts_arm.arm, ts_arm.probability))
        
        # play the arm with the best Probability
        choice = max(self.ts_arms, key=attrgetter('probability'))
        new_action = choice.arm
        print("new_action = {}".format(new_action))

        self.last_action = new_action
        return new_action

    def update_posterior(self, reward):
        print("next = {}".format(next((ts_arm for ts_arm in self.ts_arms if ts_arm.arm == self.last_action), None)))
        for ts_arm in self.ts_arms:
            if ts_arm.arm == self.last_action:
                # update the arm   Assume that <100 utility is success
                if reward < 100:
                    ts_arm.successes += 1
                else:
                    ts_arm.failures += 1

                print("ts_arm {} updated to successes = {} and failures {}".format(ts_arm.arm, ts_arm.successes, ts_arm.failures))

    def update_posterior_normal(self, reward):

        print("next = {}".format(next((ts_arm for ts_arm in self.ts_arms if ts_arm.arm == self.last_action), None)))
        for ts_arm in self.ts_arms:
            if ts_arm.arm == self.last_action:
                ts_arm.rewards.append(reward)
                ts_arm.calculate_new_mu_and_sigma()

                print("ts_arm {} updated to mu = {} and sigma {}".format(ts_arm.arm, ts_arm.mu_0, ts_arm.sigma_0))

    def update_posterior_normal_from_gamma(self, reward):
        for ts_arm in self.ts_arms:
            if ts_arm.arm == self.last_action:
                ts_arm.rewards.append(reward)  # add new observation for this arm
                ts_arm.update_normal_posterior_with_normal_gamma_prior()

                print("ts_arm {} updated to mu = {} and sigma {}".format(ts_arm.arm, ts_arm.mu_0, ts_arm.sigma_0))


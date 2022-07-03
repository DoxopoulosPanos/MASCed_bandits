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
    def __init__(self, prior_formula, posterior_formula):
        super().__init__("TS-" + prior_formula + "_" + posterior_formula )
        self.ts_arms = []
        initial_configuration = bandit_args["initial_configuration"]
        self.game_list = []
        self.last_action = initial_configuration

        # initialise arms and probabilities
        for arm in self.arms:
            self.ts_arms.append(TS_arm(arm))

    def start_strategy(self, reward):
        self.game_list.append([reward, self.last_action])

        # update values based on reward
        self.update_posterior(reward)


        # iterate for each arm
        for ts_arm in self.ts_arms:
            ts_arm.sample()

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

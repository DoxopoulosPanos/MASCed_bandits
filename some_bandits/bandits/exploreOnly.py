
import numpy as np
import time
from some_bandits.utilities import calculate_utility, convert_conf
from some_bandits.bandit_options import bandit_args
from some_bandits.bandits.Bandit import Bandit

DECAY_RATE = 1

REWARD = 0
ACTION = 1
class exploreOnly(Bandit):
    def __init__(self, formula):
        super().__init__("exploreOnly")

        initial_configuration = bandit_args["initial_configuration"]
        self.game_list = []
        self.last_action = initial_configuration

        self.epsilon = float(formula)

    def start_strategy(self, reward):
        self.game_list.append([reward, self.last_action])

        new_action = self.arms[len(self.game_list) + 1]

        self.last_action = new_action
        return new_action



    

    



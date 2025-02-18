import numpy as np
from random import sample
from some_bandits.bandits.Expert import Expert
from some_bandits.bandits.Bandit import Bandit
from some_bandits.bandit_options import bandit_args
from some_bandits.utilities import calculate_utility, convert_conf
from some_bandits.bandits.EXP3 import EXP3
from some_bandits.bandits.EwS import EwS
ACTION = 0
REWARD = 1
N_K = 2

TOTAL_ROUNDS = 1#round(3000 / 60)

class EXP4(Bandit):
    def __init__(self, formula):
        super().__init__("EXP4-" + formula)
        
        self.num_exps = bandit_args["number_of_experts"]
        
        self.expert = self.expert_to_class(bandit_args["expert"])

        self.distribution = None
        
        self.eta = 0.1#np.sqrt( (2 * np.log(self.num_exps)) / (len(self.arms) * TOTAL_ROUNDS))

        self.experts = []

        self.knowledge = None
        self.previous_expert = 0 #this can be any expert but needs to be specified.
       
        self.last_action = bandit_args["initial_configuration"]

        self.distribution = [1.0/self.num_exps] * self.num_exps

        for i in range(self.num_exps):
            exp_instance = self.expert("FH")
            #exp_instance.eta = 0.1#np.sqrt( np.log(len(self.arms))/ (TOTAL_ROUNDS * len(self.arms))   )

            if(bandit_args["preload_knowledge"]):
                
                if(isinstance(exp_instance,EwS)):
                    exp_instance.weights = bandit_args["expert_preknowledge"][i][0]
                    exp_instance.arm_reward_pairs = bandit_args["expert_preknowledge"][i][1]
                else:
                    exp_instance.weights = bandit_args["expert_preknowledge"][i]
            exp_instance.distr_func()
            self.experts.append(exp_instance)
        #print("BOO")
        

    
    def start_strategy(self, reward):
        #print("BAA")
        #self.expert_status()
        #print("My distribution is " + str(self.distribution))

        experts_matrix = np.matrix([expert.distribution for expert in self.experts])
        dist_over_arms = self.distribution * experts_matrix

        dist_over_arms = list(np.array(dist_over_arms).flatten())

        approx_arm_rewards = []
        for i in range(len(self.arms)):
            approx_rew = 1 - ( (1 if self.arms[i] == self.last_action else 0) /dist_over_arms[i]) * (1 - reward)
            approx_arm_rewards.append(approx_rew)
        
        experts_weights = list(np.array(np.matmul(experts_matrix,approx_arm_rewards)).flatten())
   
        #self.experts[self.previous_expert].propagate_reward(reward, self.arms.index(self.last_action)) this makes the expert learn


        sum_prev_weights = sum([ np.exp(self.eta * experts_weights[j])* self.distribution[j] for j in range(self.num_exps)])

        for odd_index, expert_odd in enumerate(self.distribution):
            adjusted_weight = np.exp(self.eta * experts_weights[odd_index])
            
            self.distribution[odd_index] = ((adjusted_weight * expert_odd) / sum_prev_weights)

 
        expert_choice = np.random.choice(np.arange(0, self.num_exps), p= self.distribution) #first choose the expert

        chosen_action = self.arms[self.experts[expert_choice].sample_action()]  #get action from that expert
        
        self.last_action = chosen_action
        self.previous_expert = expert_choice
       
        return chosen_action
    
    def expert_to_class(self, choice):
        funcs = {
                "EXP3": EXP3,
                "EwS" : EwS
            }
            
        func = funcs.get(choice)
        ##print(func.__doc__)
        return func

    def expert_status(self):
        for i, expert in enumerate(self.experts):
            print("-----EXPERT" + str(i) +"---")
            print('Distribution: ' + str(expert.distribution))
            print('Weights: ' + str(expert.weights))
            print("-----END EXPERT---")



import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys 
import re
import csv
from itertools import groupby
import glob
import collections
from statistics import mean
"""
This script plots vertical frequency bars for one bandit experiment.

Give -c as to load the experiment data from .csv files
"""

NUM_BARS = 2
BOUNDS = (0,35)
DIFFERENCE = False
PROPORTIONAL = False
#ideally when generalized, I want two bars to be placed in the 2/6 and 4/6 slots of a plot 
arms = [(1, 1.0), (2, 1.0), (3, 1.0)] #[(1, 1.0), (2, 1.0), (3, 1.0)]




def arms_rewards_fromCSV(filepath):
    configs = []
    rewards = []
    with open(filepath, newline='') as csvfile:
        utildimser_reader = csv.reader(csvfile)

        next(utildimser_reader)        
        for row in utildimser_reader:

            try:
                #print(row)
                configs.append((int(float(row[3])), round(float(row[2]), 2)))
                rewards.append(float(row[1]))
            except Exception as e:
                print(e)
                print("Exception in file " + filepath)
                print("Row is " + str(row))


    return configs, rewards



num_folders = (int(sys.argv[1]))

folder_names = []
plot_data = []
best_arm_input = input("best arm: e.g. 4,1.0 >").split(',')
best_arm = (int(best_arm_input[0]),float(best_arm_input[1]))

for i in range(num_folders):
    files = None
    arm_choices = []
    rewards = []
    
    folder = sys.argv[2+i]


    if(folder[-1] != "/"): folder+= "/"

    files = glob.glob(folder + "*.csv")

    folder_names.append(input("Name of line from folder " + str(folder)))
    bandit_rounds = []
    all_bandit_arms = []
    for j, filee in enumerate(files):
        arm, rew = arms_rewards_fromCSV(filee) #all the arms from run i
        bandit_rewards = []
        bandit_arms = []
        for i, a in enumerate(arm):

            if(a[1] < 1):
                #print("skipped a cleaning window")
                continue
            else:
                #print("this happened")
                bandit_arms.append(a)
                bandit_rewards.append(rew[i])
        #print(bandit_rewards)

        #print(bandit_arms)
        #xit(0)
        all_bandit_arms.append(bandit_arms)
        bandit_rounds.append(len(bandit_rewards))
        
    least_rounds = min(bandit_rounds)

  
    #print(bandit_arms)
    #exit(1)
    convergence_ratio = []
    #print("bandit arms")
    #print(bandit_arms)
    phase_len = 5
    for b_round in range(0,least_rounds-(phase_len-1),phase_len):

        ratio_in_phase = []
        for k in range(len(files)): #30 runs
            #print(k)
            phase_choices = all_bandit_arms[k][b_round:b_round+phase_len]
            opt_freq = float(collections.Counter(phase_choices)[best_arm])/len(phase_choices)
            ratio_in_phase.append(opt_freq)
        #print("ratio in phase")
        #print(ratio_in_phase)
        convergence_ratio.append(mean(ratio_in_phase))
    
    #print(convergence_ratio)
    plot_data.append(convergence_ratio)
    
convergence_boundary = float(input("convegence boundary > "))
colors = ['k', 'c', 'm', 'b']
print("The shortest run is " + str(len(min(plot_data, key = lambda k: len(k)))))

shortest_rounds = int(input("How many rounds should the plot include? > "))
print("plot_data")
print(plot_data)
for i, datum in enumerate(plot_data):

    plt.plot(datum[0:shortest_rounds], color = colors[i], label=folder_names[i])
    last_bad = 0
    for j, phasum in enumerate(datum):
        if phasum <= convergence_boundary:
            last_bad = j
    plt.vlines(last_bad,0.0,convergence_boundary, color=colors[i], linestyles='dashed',label='Conv. Point ' + folder_names[i] )
    #print((last_bad,0.0,convergence_boundary, colors[i], 'dashed','Conv. Point ' + folder_names[i] ))


#longest = len(max(plot_data, key = lambda k: len(k)))

plt.hlines(convergence_boundary, 0, shortest_rounds-1, color ='b', label='conv. boundary')
plt.legend(fontsize='small', loc='lower right')
plt.xlabel('bandit rounds')
plt.ylabel('convergence factor')
plt.xticks(np.arange(0,shortest_rounds+1,3))
plt.savefig(input("plot name > ") + ".pdf")
plt.cla()



            

        
    


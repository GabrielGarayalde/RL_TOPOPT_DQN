"""
Deep Learning Reinforcement Tutorial: Deep Q Network (DQN) = Combination of Deep Learning and Q-Learning Tutorial

This file contains driver code that imports DeepQLearning class developed in the file "functions_final"
 
The class DeepQLearning implements the Deep Q Network (DQN) Reinforcement Learning Algorithm.
The implementation is based on the OpenAI Gym Cart Pole environment and TensorFlow (Keras) machine learning library

The webpage explaining the codes and the main idea of the DQN is given here:

https://aleksandarhaber.com/deep-q-networks-dqn-in-python-from-scratch-by-using-openai-gym-and-tensorflow-reinforcement-learning-tutorial/


Author: Aleksandar Haber 
Date: February 2023

Tested on:

tensorboard==2.11.2
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow==2.11.0
tensorflow-estimator==2.11.0
tensorflow-intel==2.11.0
tensorflow-io-gcs-filesystem==0.30.0

keras==2.11.0

gym==0.26.2

"""
# import the class
from functions_DQN import DeepQLearning
from Methods_RL import Mesh
from Classes_RL import Node

# # classical gym 
# import gym
# # instead of gym, import gymnasium 
# #import gymnasium as gym

# # create environment
# env=gym.make('CartPole-v1')

from truss import TrussEnv

from math import ceil
from math import cos,acos
import numpy as np

# --- MATERIAL ATTRIBUTES ---- #

Emod = 10000#210e9
Area = 1#38.77e-4
density = 0#7800

material = {"E": Emod, "Area": Area, "Density": density}

# --- LEARNING HYPERPARAMETERS --- #

VolumeMax = 240#250
num_episodes = 3000#9000

learning_rate = 0.1
discount_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01 #if we decrease it, will learn slower

wavelength1 = ceil(num_episodes*0.5)
wavelength2 = ceil(num_episodes*0.75)

B_a = acos(0.1)/wavelength2
B_e = acos(0.01/0.9)/wavelength1
# B_a = 2.65*1e-4
# B_e = 3.1*1e-4

A_e = 0.9
A_a = 1

# Negative reward for unallowed actions
penalty=-10

# --- DOMAIN ATTRIBUTES --- #
# enter domain dimension: x,y
x,y = 20, 40
#enter number of nodes, (rows),(columns)
xm,yn = 5, 3
# compute spacing between nodes
xs = int(x/(yn-1))
ys = int(y/(xm-1))

#Spring coeff
springCoeff = 0.1*Emod*Area/np.min((xs,ys))

springValues = np.array([])

spring = {
    "values": springValues,
    "coeff": springCoeff}

#grid world, allowed domain
grid = Mesh(xm,yn,ys,xs)
totalnodes = xm*yn

initialState = [[0, 14], [0, 8], [2, 8], [8, 14]]
# --- NODES LIST HOLDING INSTANCES OF NODE CLASS FOR WHOLE GRID --- #
Nodes = []
for i in range(len(grid)):
    node = Node(grid[i][0], grid[i][1])
    Nodes.append(node)

# INSERT THE BC'S BY SELECTING THE NODE AND DOF DIR'N AND SETTING TO FALSE
Nodes[0].freeDOF_x = False
Nodes[0].freeDOF_y = False
Nodes[2].freeDOF_x = False
Nodes[2].freeDOF_y = False

#INSERT THE NODE FORCES
Nodes[8].force_x = 10
Nodes[14].force_x = 10



env = TrussEnv(initialState, Nodes, grid, material, spring)

# (currentState,_)= env.reset()

# select the parameters
gamma=1
# probability parameter for the epsilon-greedy approach
epsilon=0.1
# number of training episodes
# NOTE HERE THAT AFTER CERTAIN NUMBERS OF EPISODES, WHEN THE PARAMTERS ARE LEARNED
# THE EPISODE WILL BE LONG, AT THAT POINT YOU CAN STOP THE TRAINING PROCESS BY PRESSING CTRL+C
# DO NOT WORRY, THE PARAMETERS WILL BE MEMORIZED
numberEpisodes=1000

# create an object
LearningQDeep=DeepQLearning(env,gamma,epsilon,numberEpisodes)
# run the learning process
LearningQDeep.trainingEpisodes()
# get the obtained rewards in every episode
LearningQDeep.sumRewardsEpisode

#  summarize the model
LearningQDeep.mainNetwork.summary()
# save the model, this is important, since it takes long time to train the model 
# and we will need model in another file to visualize the trained model performance
LearningQDeep.mainNetwork.save("trained_model_temp.h5")



"""

Environment Imports:

"""

import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

# Import Lunar Landing Environment
import gym
env = gym.make('LunarLander-v2')

import torch.optim as optim
def make_dqn():
    model = nn.Sequential(nn.Linear(64,),
                          nn.ReLU(inplace = True),
                          nn.Linear(64,),
                          nn.ReLU(inplace = True))
    loss_function = nn.MSELoss # define MSE as the loss function
    optimizer = optim.Adam(nn.parameters(), lr = 0.0001)

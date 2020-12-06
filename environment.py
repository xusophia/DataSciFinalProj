import numpy as np
import torch


def add_input_noise(state):
    """
    Adding sensor input noise to location values of state
    """
    noise = np.random.normal(loc=0,scale=0.05,size=3)
    state[0] += noise[0]
    state[1] += noise[1]
    state[4] += noise[2]
    return state


def add_action_noise(action):
    """
    Adding noise to the action (engine) of the lander
    """
    if np.random.randint(0, 10) < 2:
        return torch.tensor([0])
    else:
        return action

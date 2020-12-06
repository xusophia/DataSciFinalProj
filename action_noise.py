import numpy as np


def action_noise(action):
    if np.random.randint(0, 10) < 2:
        return 0
    else:
        return action

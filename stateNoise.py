import numpy as np

def addNoise(state):
    #adding noise to location values of state
    noise = np.random.normal(loc=0,scale=0.05,size=3)
    state[0] += noise[0]
    state[1] += noise[1]
    state[4] += noise[2]
    return state
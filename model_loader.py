import gym
import random
from keras import models
import matplotlib.pyplot as plt
import numpy as np
import torch

env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)

# my_model = models.load_model("def_DQN.h5")
my_model = torch.load("VanillaDQN.pt")

# state = env.reset()
# state = np.reshape(state, (1, 8))
# score = 0
# max_steps = 3000
# for i in range(max_steps):
#     act_values = my_model.predict(state)
#     action = np.argmax(act_values[0])
#     env.render()
#     next_state, reward, done, _ = env.step(action)
#     score += reward
#     next_state = np.reshape(next_state, (1, 8))
#     # agent.remember(state, action, reward, next_state, done)
#     state = next_state
    # agent.replay()
    # if done:
    #     print("episode: {}/{}, score: {}".format(e, episode, score))
    #     break

state = env.reset().astype(np.float32)
reward_ep, done = 0, False
while not done:
    env.render() # Comment this if you do not want rendering
    state_tensor = torch.from_numpy(state)
    action = my_model.get_action(state_tensor)
    next_state, reward, done, info = env.step(action)
    next_state = next_state.astype(np.float32)
    reward_ep += reward
    # my_model.insert(state, action, reward, next_state, done)
    state = next_state

print(reward_ep)
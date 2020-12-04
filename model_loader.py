import gym
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from stateNoise import addNoise
from VanillaDqn import DQN
env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)

epsilon = 0.05
gamma = 0.99
buffer_size = 500000
counter = 0
update_t = 5
batch_size = 32
lr = 1e-3
network = DQN(env, lr = lr, gamma = gamma, epsilon = epsilon, buffer_size =buffer_size)
# my_model = models.load_model("def_DQN.h5")
my_model = torch.load("saved/lunarlanderVanilla.pt")
network.load_state_dict(my_model)
network.eval()
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

for i in range(10):
    state = env.reset().astype(np.float32)
    reward_ep, done = 0, False
    while not done:
        env.render() # Comment this if you do not want rendering
        state_tensor = torch.from_numpy(state)
        action = network.get_action(state_tensor)
        next_state, reward, done, info = env.step(action)
        next_state = next_state.astype(np.float32)
        next_state = addNoise(next_state)
        reward_ep += reward
        # my_model.insert(state, action, reward, next_state, done)
        state = next_state

    print(reward_ep)
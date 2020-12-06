'''Dueling DQN training in clear environment'''
import gym
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from stateNoise import addNoise
import VanillaDqn
import duelingDqn
env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)


gamma = 0.99
buffer_size = 500000
counter = 0
update_t = 5
batch_size = 32
lr = 1e-3
# duel_network = duelingDqn.DQN(env, lr = lr, gamma = gamma, epsilon = 0.03, buffer_size =buffer_size)
# duel_noise_network = duelingDqn.DQN(env, lr = lr, gamma = gamma, epsilon = 0.01, buffer_size =buffer_size)
# # my_model = models.load_model("def_DQN.h5")
# duel_network.load_state_dict(torch.load("saved/lunarlanderDuel.pt"))
# duel_network.eval()
# duel_noise_network.load_state_dict(torch.load("saved/lunarlander_Duel_noise_0.5.pt"))
# duel_noise_network.eval()
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


def run_model(network, noise = False):
    reward_list = []
    for i in range(10):
        state = env.reset().astype(np.float32)
        reward_ep, done = 0, False
        while not done:
            env.render() # Comment this if you do not want rendering
            state_tensor = torch.from_numpy(state)
            action = network.get_action(state_tensor)
            next_state, reward, done, info = env.step(action)
            next_state = next_state.astype(np.float32)
            if noise:
                next_state = addNoise(next_state)
            reward_ep += reward
            # my_model.insert(state, action, reward, next_state, done)
            state = next_state

        reward_list.append(reward_ep)
    return reward_list

# clear_duel =run_model(duel_network,noise=True)
# noise_duel = run_model(duel_noise_network,noise=True)


# fig = plt.figure(figsize=(20,10))
# plt.plot([i for i in range(len(clear_duel))], clear_duel, 'y', label="Duel trained without noise")
# plt.xlabel("Episodes")
# plt.ylabel("Rewards")
#
# plt.plot([i for i in range(len(noise_duel))], noise_duel, 'b', label="duel trained with noise")
# plt.legend(loc="upper left")
# plt.savefig('noise_results/Trained_duel_with_noise_0.5.png')
#
# duel_in_noise = np.asarray(noise_duel)
# duel_without_noise = np.asarray(clear_duel)
# np.savetxt("rewards/duel_trained_with_noise_0.5.csv",duel_in_noise,delimiter=",")
# np.savetxt("rewards/duel_trained_without_noise_0.5.csv",duel_without_noise,delimiter=",")

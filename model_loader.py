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
van_network = VanillaDqn.DQN(env, lr = lr, gamma = gamma, epsilon = 0.03, buffer_size =buffer_size)
duel_network = duelingDqn.DQN(env, lr = lr, gamma = gamma, epsilon = 0.06, buffer_size =buffer_size)
# my_model = models.load_model("def_DQN.h5")
van_network.load_state_dict(torch.load("saved/lunarlanderVanilla.pt"))
van_network.eval()
duel_network.load_state_dict(torch.load("saved/lunarlanderDuel.pt"))
duel_network.eval()
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
    for i in range(100):
        state = env.reset().astype(np.float32)
        reward_ep, done = 0, False
        while not done:
            #env.render() # Comment this if you do not want rendering
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

clear_vanilla =run_model(van_network,noise=False)
clear_duel = run_model(duel_network,noise=False)
noise_vanilla = run_model(van_network,noise=True)
noise_duel = run_model(duel_network,noise=True)

fig = plt.figure(figsize=(20,10))
plt.plot([i for i in range(len(clear_vanilla))], clear_vanilla, 'y', label="van in clear")
plt.xlabel("Episodes")
plt.ylabel("Rewards")


plt.plot([i for i in range(len(clear_duel))], clear_duel, 'g', label="duel in clear")


plt.plot([i for i in range(len(noise_vanilla))], noise_vanilla, 'r', label="van in Noise")


plt.plot([i for i in range(len(noise_duel))], noise_duel, 'b', label="duel in Noise")
plt.legend(loc="upper left")
plt.savefig('noise_results/joint_plot_0.05.png')

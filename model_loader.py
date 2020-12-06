import gym
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from VanillaDqn import DQN
from DoubleDqn import DoubleDQN
from duelingDqn import DuelingDQN

env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)

gamma = 0.99
buffer_size = 500000
counter = 0
update_t = 5
batch_size = 32
lr = 1e-3
# van_network = DQN(env, lr=lr, gamma=gamma, epsilon=0.03, buffer_size=buffer_size)
# double_network = DoubleDQN(env, lr=lr, gamma=gamma, epsilon=0.06, buffer_size=buffer_size)
dueling_network = DuelingDQN(env, lr=lr, gamma=gamma, epsilon=0.06, buffer_size=buffer_size)
# my_model = models.load_model("def_DQN.h5")
# van_network.load_state_dict(torch.load("saved/lunarlanderVanilla.pt"))
# van_network.eval()
# double_network.load_state_dict(torch.load("saved/lunarlanderDouble.pt"))
# double_network.eval()
dueling_network.load_state_dict(torch.load("saved/lunarlanderDuel.pt"))
dueling_network.eval()


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


def run_model(network, noise=False):
    reward_list = []
    for i in range(100):
        state = env.reset().astype(np.float32)
        reward_ep, done = 0, False
        while not done:
            # env.render() # Comment this if you do not want rendering
            state_tensor = torch.from_numpy(state)
            action = network.get_action(state_tensor)
            if noise:
                if np.random.randint(0, 10) < 2:
                    action = 0
            next_state, reward, done, info = env.step(action)
            next_state = next_state.astype(np.float32)
            reward_ep += reward
            # my_model.insert(state, action, reward, next_state, done)
            state = next_state

        if i % 50 == 0 and i > 1:
            print(f'Episode {i}/100. Epsilon: {network.epsilon:.3f}.')
        reward_list.append(reward_ep)
    return reward_list


# ideal_vanilla = run_model(van_network, noise=False)
# ideal_double = run_model(double_network, noise=False)
# noise_vanilla = run_model(van_network, noise=True)
# noise_double = run_model(double_network, noise=True)

ideal_dueling = run_model(dueling_network, noise=False)
noise_dueling = run_model(dueling_network, noise=True)

fig = plt.figure(figsize=(20, 10))
# plt.plot([i for i in range(len(ideal_vanilla))], ideal_vanilla, 'y', label="ideal vanilla")
plt.xlabel("Episodes")
plt.ylabel("Rewards")

# plt.plot([i for i in range(len(ideal_double))], ideal_double, 'g', label="ideal double")

# plt.plot([i for i in range(len(noise_vanilla))], noise_vanilla, 'r', label="vanilla in noise")

# plt.plot([i for i in range(len(noise_double))], noise_double, 'b', label="double in noise")

plt.plot([i for i in range(len(ideal_dueling))], ideal_dueling, 'g', label="ideal double")

plt.plot([i for i in range(len(noise_dueling))], noise_dueling, 'r', label="vanilla in noise")
plt.legend(loc="upper left")
plt.savefig('results/idealInNoisyDueling.png')

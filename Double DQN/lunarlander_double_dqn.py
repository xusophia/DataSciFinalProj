import gym
from VanillaDqn import DQN
from DoubleDqn import DoubleDQN
import numpy as np
import torch

from gym.wrappers.monitoring.video_recorder import VideoRecorder
import matplotlib.pyplot as plt

env = gym.make('LunarLander-v2')
# recorder = VideoRecorder(env, path='results/vanilladqn.mp4')
episodes = 500
epsilon = 1.0
gamma = 0.99
buffer_size = 500000
counter = 0
update_t = 5
batch_size = 32
lr = 1e-3
network = DoubleDQN(env, lr=lr, gamma=gamma, epsilon=epsilon, buffer_size=buffer_size)
reward_list_ep = []
reward_last_100_eps = []

for episode in range(episodes + 1):
    state = env.reset().astype(np.float32)
    reward_ep, done = 0, False

    while not done:
        # env.render()  # Comment this if you do not want rendering
        # env.unwrapped.render() # Comment this if you do not want rendering
        # recorder.capture_frame()
        state_tensor = torch.from_numpy(state)
        action = network.get_action(state_tensor)
        next_state, reward, done, info = env.step(action)
        next_state = next_state.astype(np.float32)
        reward_ep += reward
        network.insert(state, action, reward, next_state, done)
        state = next_state

        counter += 1
        counter %= update_t

        if len(network.buffer) > batch_size:  # update weights every 5 steps
            states, actions, rewards, next_states, dones = network.sample_buffer(batch_size)
            loss = network.train_batch(states, actions, rewards, next_states, dones)

    if episode < 900:
        network.epsilon *= network.epsilon_decay
    reward_list_ep.append(reward_ep)

    if len(reward_last_100_eps) == 100:
        reward_last_100_eps = reward_last_100_eps[1:]
    reward_last_100_eps.append(reward_ep)

    if episode % 50 == 0 and episode > 1:
        print(f'Episode {episode}/{episodes}. Epsilon: {network.epsilon:.3f}.'
              f' Reward in the last 100 episodes: {np.mean(reward_last_100_eps):.2f}')
    last_rewards_mean = np.mean(reward_last_100_eps)
    if last_rewards_mean > 200:
        break
env.close()

# PATH1 = 'saved/lunarlanderDouble.pt'
# torch.save(network.state_dict(), PATH1)
# fig = plt.figure(figsize=(20, 10))
plt.scatter([i for i in range(len(reward_list_ep))], reward_list_ep)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.savefig('DoubleDQNscatter - Wind_divided_by_6.png')
plt.plot([i for i in range(len(reward_list_ep))], reward_list_ep)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.savefig('DoubleDQNscatter - Wind_divided_by_6_plt')

# fig = plt.figure(figsize=(20, 10))
# plt.scatter([i for i in range(len(reward_list_ep))], reward_list_ep)
# plt.xlabel("Episodes")
# plt.ylabel("Rewards")
# plt.savefig('results/vanillaDQNscatter.png')
# plt.plot([i for i in range(len(reward_list_ep))], reward_list_ep)
# plt.xlabel("Episodes")
# plt.ylabel("Rewards")
# plt.savefig('results/vanillaDQNplot.png')

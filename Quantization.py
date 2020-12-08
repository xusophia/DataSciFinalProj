# File for Quantization of Models

# import torch.quantization
import torch.nn as nn
from vanilla import DQN
import gym
import numpy as np
import os
env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)


gamma = 0.99
buffer_size = 500000
counter = 0
update_t = 5
batch_size = 32
lr = 1e-3

episodes = 1000
epsilon = 1.0
gamma = 0.99
buffer_size = 500000
counter = 0
update_t = 5
batch_size = 32
lr = 1e-3
network = DQN(env, lr = lr, gamma = gamma, epsilon = epsilon, buffer_size =buffer_size)
reward_list_ep = []
reward_last_100_eps = []
import tensorflow as tf
for episode in range(episodes+1):
    state = env.reset().astype(np.int8)
    reward_ep, done = 0, False

    while not done:
        env.unwrapped.render() # Comment this if you do not want rendering
        #recorder.capture_frame()
        state_tensor = torch.from_numpy(state)
        # state_tensor = state_tensor.byte()
        state_tensor = state_tensor.type(dtype = int8)
        print(type(state_tensor))
        # x = x.type(torch.ByteTensor)
        action = network.get_action(state_tensor)
        #modify here so that action returned is taken only 80% of time and 20% action is set to 0(or the value that means nothing)
        next_state, reward, done, info = env.step(action)
        next_state = next_state.astype(np.int8)
        reward_ep += reward
        network.insert(state.astype(np.float32), action, reward, next_state.astype(np.float32), done)
        state = next_state

        counter += 1
        counter %= update_t

        if len(network.buffer) > batch_size and counter==0: #update weights every 5 steps
            states, actions, rewards, next_states, dones = network.sample_buffer(batch_size)
            loss = network.train_batch(states, actions, rewards, next_states, dones)

    if episode < 900:
        network.epsilon*=network.epsilon_decay
    reward_list_ep.append(reward_ep)

    if len(reward_last_100_eps) == 100:
        reward_last_100_eps = reward_last_100_eps[1:]
    reward_last_100_eps.append(reward_ep)

    if episode % 50 == 0 and episode>1:
        print(f'Episode {episode}/{episodes}. Epsilon: {network.epsilon:.3f}.'
              f' Reward in the last 100 episodes: {np.mean(reward_last_100_eps):.2f}')
    last_rewards_mean = np.mean(reward_last_100_eps)
    if last_rewards_mean>200:
        break
env.close()
quantized_model = torch.quantization.quantize_dynamic(network, {nn.Linear}, dtype=torch.qint8)
print(quantized_model)
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print('Size of Original Vanilla DQN:')
print_size_of_model(network)
print('Size of Quantized DQN:')
print_size_of_model(quantized_model)

# from model_loader import run_model
# result = run_model(quantized_model)

env.close()
for episode in range(episodes+1):
    state = env.reset().astype(np.float32)
    reward_ep, done = 0, False

    while not done:
        #env.unwrapped.render() # Comment this if you do not want rendering
        #recorder.capture_frame()
        state_tensor = torch.from_numpy(state)
        action = quantized_model.get_action(state_tensor)
        #modify here so that action returned is taken only 80% of time and 20% action is set to 0(or the value that means nothing)
        next_state, reward, done, info = env.step(action)
        next_state = next_state.astype(np.float32)
        reward_ep += reward
        quantized_model.insert(state, action, reward, next_state, done)
        state = next_state

        counter += 1
        counter %= update_t

        if len(quantized_model.buffer) > batch_size and counter==0: #update weights every 5 steps
            states, actions, rewards, next_states, dones = quantized_model.sample_buffer(batch_size)
            loss = quantized_model.train_batch(states, actions, rewards, next_states, dones)

    if episode < 900:
        quantized_model.epsilon*=quantized_model.epsilon_decay
    reward_list_ep.append(reward_ep)

    if len(reward_last_100_eps) == 100:
        reward_last_100_eps = reward_last_100_eps[1:]
    reward_last_100_eps.append(reward_ep)

    if episode % 50 == 0 and episode>1:
        print(f'Episode {episode}/{episodes}. Epsilon: {quantized_model.epsilon:.3f}.'
              f' Reward in the last 100 episodes: {np.mean(reward_last_100_eps):.2f}')
    last_rewards_mean = np.mean(reward_last_100_eps)
    if last_rewards_mean>200:
        break

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(20,10))
plt.scatter([i for i in range(len(reward_list_ep))], reward_list_ep)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.savefig('Quantized.png')
plt.plot([i for i in range(len(reward_list_ep))], reward_list_ep)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.savefig('Quantized_Plot.png')



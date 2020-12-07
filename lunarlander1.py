import gym
from VanillaDqn import DQN
import numpy as np
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import matplotlib.pyplot as plt
env = gym.make('LunarLander-v2')

# import torchbackend

from model_loader import run_model

# recorder = VideoRecorder(env, path='results/vanilladqn.mp4')
episodes = 1000
epsilon = 1.0
gamma = 0.99
buffer_size = 500000
counter = 0
update_t = 5
batch_size = 32
lr = 1e-3
network_fp32 = DQN(env, lr = lr, gamma = gamma, epsilon = epsilon, buffer_size =buffer_size)

USE_FBGEMM=1
# Prepare a Quantitized model

# Prepare the modelL ensure config and the engine used for quantized computations match the backend
# on which the model will be executed
qconfig = torch.quantization.get_default_qconfig('fbgemm')

qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
# torch.backends.quantized.engine = 'qnnpack'
# torch.backends.quantized.engine='fbgemm'


print(torch.backends.quantized.supported_engines)

reward_list_ep = []
reward_last_100_eps = []
# for episode in range(episodes+1):
#     state = env.reset().astype(np.float32)
#     reward_ep, done = 0, False
#
#     while not done:
#         env.unwrapped.render() # Comment this if you do not want rendering
#         #recorder.capture_frame()
#         state_tensor = torch.from_numpy(state)
#         action = network_fp32.get_action(state_tensor)
#         next_state, reward, done, info = env.step(action)
#         next_state = next_state.astype(np.float32)
#         reward_ep += reward
#         network_fp32.insert(state, action, reward, next_state, done)
#         state = next_state
#
#         counter += 1
#         counter %= update_t
#
#         if len(network_fp32.buffer) > batch_size and counter==0: #update weights every 5 steps
#             states, actions, rewards, next_states, dones = network_fp32.sample_buffer(batch_size)
#             loss = network_fp32.train_batch(states, actions, rewards, next_states, dones)
#
#     if episode < 900:
#         network_fp32.epsilon*=network_fp32.epsilon_decay
#     reward_list_ep.append(reward_ep)
#
#     if len(reward_last_100_eps) == 100:
#         reward_last_100_eps = reward_last_100_eps[1:]
#     reward_last_100_eps.append(reward_ep)
#
#     if episode % 50 == 0 and episode>1:
#         print(f'Episode {episode}/{episodes}. Epsilon: {network_fp32.epsilon:.3f}.'
#               f' Reward in the last 100 episodes: {np.mean(reward_last_100_eps):.2f}')
#     last_rewards_mean = np.mean(reward_last_100_eps)
#     if last_rewards_mean>200:
#         break

network_fp32.qconfig  = torch.quantization.get_default_qat_qconfig('fbgemm')
# network_fp32_fused = torch.quantization.fuse_modules(network_fp32, ['relu'])
# Prepare the model for QAT. This inserts observers and fake_quants in
# the model that will observe weight and activation tensors during calibration.
network_fp32_prepared = torch.quantization.prepare(network_fp32)


# run the training loop again
# for episode in range(episodes+1):
#     state = env.reset().astype(np.float32)
#     reward_ep, done = 0, False
#
#     while not done:
#         env.unwrapped.render() # Comment this if you do not want rendering
#         #recorder.capture_frame()
#         state_tensor = torch.from_numpy(state)
#         action = network_fp32_prepared.get_action(state_tensor)
#         next_state, reward, done, info = env.step(action)
#         next_state = next_state.astype(np.float32)
#         reward_ep += reward
#         network_fp32_prepared.insert(state, action, reward, next_state, done)
#         state = next_state
#
#         counter += 1
#         counter %= update_t
#
#         if len(network_fp32_prepared.buffer) > batch_size and counter==0: #update weights every 5 steps
#             states, actions, rewards, next_states, dones = network_fp32_prepared.sample_buffer(batch_size)
#             loss = network_fp32_prepared.train_batch(states, actions, rewards, next_states, dones)
#
#     if episode < 900:
#         network_fp32_prepared.epsilon*=network_fp32_prepared.epsilon_decay
#     reward_list_ep.append(reward_ep)
#
#     if len(reward_last_100_eps) == 100:
#         reward_last_100_eps = reward_last_100_eps[1:]
#     reward_last_100_eps.append(reward_ep)
#
#     if episode % 50 == 0 and episode>1:
#         print(f'Episode {episode}/{episodes}. Epsilon: {network_fp32_prepared.epsilon:.3f}.'
#               f' Reward in the last 100 episodes: {np.mean(reward_last_100_eps):.2f}')
#     last_rewards_mean = np.mean(reward_last_100_eps)
#     if last_rewards_mean>200:
#         break

# Convert the observed model to a quantized model. This does several things:
# quantizes the weights, computes and stores the scale and bias value to be
# used with each activation tensor, fuses modules where appropriate,
# and replaces key operators with quantized implementations.
network_fp32_prepared.eval()
network_int8 = torch.quantization.convert(network_fp32_prepared, inplace = True)

# run the model, relevant calculations will happen in int8
from model_loader import run_model
result = run_model(network_int8)

env.close()

fig = plt.figure(figsize=(20,10))
plt.scatter([i for i in range(len(reward_list_ep))], reward_list_ep)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
# plt.savefig('results/vanillaDQNscatter.png')
plt.plot([i for i in range(len(reward_list_ep))], reward_list_ep)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
# plt.savefig('results/vanillaDQNplot.png')

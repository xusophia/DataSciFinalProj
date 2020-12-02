import gym
from VanillaDqn import DQN
import numpy as np
import torch
env = gym.make('LunarLander-v2')
episodes = 10000
epsilon = 1.0
gamma = 0.99
buffer_size = 500000
counter = 0
update_t = 5
batch_size = 64
lr = 0.001
network = DQN(env, lr=lr, gamma=gamma, epsilon=epsilon, buffer_size=buffer_size)
# network = torch.load("VanillaDQN_base.pt")
reward_list_ep = []
reward_last_100_eps = []

for episode in range(episodes+1):
    state = env.reset().astype(np.float32)
    reward_ep, done = 0, False
    while not done:
        env.render() # Comment this if you do not want rendering
        state_tensor = torch.from_numpy(state)
        action = network.get_action(state_tensor)
        next_state, reward, done, info = env.step(action)
        next_state = next_state.astype(np.float32)
        reward_ep += reward
        network.insert(state, action, reward, next_state, done)
        state = next_state

        counter += 1
        counter %= update_t

        if len(network.buffer) > batch_size and counter==0: #update weights every 5 steps
            states, actions, rewards, next_states, dones = network.sample_buffer(batch_size)
            loss = network.train_batch(states, actions, rewards, next_states, dones)

    if episode < 900:
        network.epsilon *= network.epsilon_decay
    reward_list_ep.append(reward_ep)

    # if len(reward_last_100_eps)==100:
    #     reward_last_100_eps = reward_last_100_eps[1:0]
    reward_last_100_eps.append(reward_ep)

    # if episode % 50 == 0:
    print(f'Episode {episode}/{episodes}. Reward: {reward_ep}. Epsilon: {network.epsilon:.3f}.'
          f' Reward in the last 100 episodes: {np.mean(reward_last_100_eps[-100:]):.2f}')
    # last_rewards_mean = np.mean(reward_last_100_eps)
    # if(last_rewards_mean>200)

    is_solved = np.mean(reward_last_100_eps[-100:])
    if is_solved > 200:
        print('\n Solved! \n')
        break

torch.save(network, "VanillaDQN.pt")
env.close()

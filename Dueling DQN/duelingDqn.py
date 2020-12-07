import gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from itertools import count
import torch.nn.functional as F

#https://arxiv.org/pdf/1511.06581.pdf

class DQN(nn.Module):
    # def __init__(self):
    #     super(QNetwork, self).__init__()

    #     self.fc1 = nn.Linear(4, 64)
    #     self.relu = nn.ReLU()
    #     self.fc_value = nn.Linear(64, 256)
    #     self.fc_adv = nn.Linear(64, 256)

    #     self.value = nn.Linear(256, 1)
    #     self.adv = nn.Linear(256, 2)

    # def forward(self, state):
    #     y = self.relu(self.fc1(state))
    #     value = self.relu(self.fc_value(y))
    #     adv = self.relu(self.fc_adv(y))

    #     value = self.value(value)
    #     adv = self.adv(adv)

    #     advAverage = torch.mean(adv, dim=1, keepdim=True)
    #     Q = value + adv - advAverage

    #     return Q
    def __init__(self, env, lr=1e-4, gamma = 0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=500000):
        super(DQN, self).__init__()
        #input output dimensions
        self.state_space_dim = env.observation_space.shape[0]
        self.action_space_dim = env.action_space.n
        #our state-action function estimator
        self.fc1 = nn.Linear(8, 512)
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(512, 256)
        self.fc_adv = nn.Linear(512, 256)

        self.value_output = nn.Linear(256, 1)  #represents the state value
        self.adv_output = nn.Linear(256, 4) # 4 actions

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.env = env
        self.buffer = deque(maxlen=buffer_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()     #defining our loss function to be the MSE loss

    def forward(self, state):
        y = self.relu(self.fc1(state))
        value = self.relu(self.fc_value(y))
        adv = self.relu(self.fc_adv(y))

        value = self.value_output(value)
        adv = self.adv_output(adv)
        #Aπ(s,a) =Qπ(s,a)−Vπ(s)
        #Q(s,a;θ,α,β) =V(s;θ,β) +(A(s,a;θ,α)−1|A|∑a′A(s,a′;θ,α))
        advAverage = torch.mean(adv, dim=-1, keepdim=True)
        Q = value + adv - advAverage
        return Q

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.forward(state).data.numpy())

    def insert(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample_buffer(self, num_samples):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idx = np.random.choice(len(self.buffer), num_samples)
        for i in idx:
            elem = self.buffer[i]
            state, action, reward, next_state, done = elem
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        states = torch.as_tensor(np.array(states))
        actions = torch.as_tensor(np.array(actions, dtype=np.int64))
        rewards = torch.as_tensor(np.array(rewards, dtype=np.float32))
        next_states = torch.as_tensor(np.array(next_states))
        dones = torch.as_tensor(np.array(dones, dtype=np.float32))
        return states, actions, rewards, next_states, dones
    
    def train_batch(self, states, actions, rewards, next_states, dones):
        # Bellman equation for updates
        targets = rewards + self.gamma * self.forward(next_states).max(-1).values * (1.0-dones)
        preds = self.forward(states)
        action_masks = F.one_hot(actions, self.action_space_dim)
        preds = (preds * action_masks).sum(dim=-1)
        loss = self.loss_fn(preds, targets.detach())
        self.optimizer.zero_grad() #zero out the gradients for weights of model
        loss.backward() #compute the gradient of loss with respect to model paramenters
        self.optimizer.step()
        return loss

# class Memory(object):
#     def __init__(self, memory_size: int) -> None:
#         self.memory_size = memory_size
#         self.buffer = deque(maxlen=self.memory_size)

#     def add(self, experience) -> None:
#         self.buffer.append(experience)

#     def size(self):
#         return len(self.buffer)

#     def sample(self, batch_size: int, continuous: bool = True):
#         if batch_size > len(self.buffer):
#             batch_size = len(self.buffer)
#         if continuous:
#             rand = random.randint(0, len(self.buffer) - batch_size)
#             return [self.buffer[i] for i in range(rand, rand + batch_size)]
#         else:
#             indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
#             return [self.buffer[i] for i in indexes]

#     def clear(self):
#         self.buffer.clear()

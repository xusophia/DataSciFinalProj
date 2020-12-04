import gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from itertools import count
import torch.nn.functional as F
from torch.autograd import Variable
from naivePrioritizedBuffer import NaivePrioritizedBuffer

from torch.autograd import Variable

#https://arxiv.org/pdf/1511.06581.pdf

class DQN(nn.Module):
    def __init__(self, env, lr=1e-4, gamma = 0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=1000000):
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
        self.buffer = NaivePrioritizedBuffer(buffer_size, prob_alpha=0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()     #defining our loss function to be the MSE loss
        
        self.beta = 1#0.4
        self.beta_increment = (1-self.beta)/1000

        self.batch_size = 32

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
        #action = torch.as_tensor(np.array(action, dtype=np.int64))
        #reward = torch.as_tensor(np.array(reward, dtype=np.float32))
        #done = torch.as_tensor(np.array(done, dtype=np.float32))
        
        # target = reward + self.gamma * self.forward(torch.as_tensor(np.array(next_state))).max(-1).values * (1.0-done)
        # pred = self.forward(torch.as_tensor(np.array(state)))
        # action_mask = F.one_hot(torch.as_tensor(np.array(action, dtype=np.int64)), self.action_space_dim)
        # pred = (pred * action_mask).sum(dim=-1)
        # error = torch.abs(pred-target).data

        self.buffer.push(state, action, reward, next_state, done)

    # def sample_buffer(self, num_samples):
    #     states, actions, rewards, next_states, dones = [], [], [], [], []
    #     idx = np.random.choice(len(self.memory), num_samples)
    #     for i in idx:
    #         elem = self.memory[i]
    #         state, action, reward, next_state, done = elem
    #         states.append(np.array(state, copy=False))
    #         actions.append(np.array(action, copy=False))
    #         rewards.append(reward)
    #         next_states.append(np.array(next_state, copy=False))
    #         dones.append(done)
    #     states = torch.as_tensor(np.array(states))
    #     actions = torch.as_tensor(np.array(actions, dtype=np.int64))
    #     rewards = torch.as_tensor(np.array(rewards, dtype=np.float32))
    #     next_states = torch.as_tensor(np.array(next_states))
    #     dones = torch.as_tensor(np.array(dones, dtype=np.float32))
    #     return states, actions, rewards, next_states, dones
    
    def train_batch(self):
        state, action, reward, next_state, done, indices, weights = self.buffer.sample(self.batch_size, self.beta) 
        #mini_batch = np.array(mini_batch).transpose()
        states     = Variable(torch.FloatTensor(np.float32(state)))
        next_states = Variable(torch.FloatTensor(np.float32(next_state)))
        actions     = Variable(torch.LongTensor(action))
        rewards     = Variable(torch.FloatTensor(reward))
        dones       = Variable(torch.FloatTensor(done))
        weights    = Variable(torch.FloatTensor(weights))

        self.beta+=self.beta_increment
        # Bellman equation for updates
        targets = rewards + self.gamma * self.forward(next_states).max(-1).values * (1.0-dones)
        preds = self.forward(states)
        action_masks = F.one_hot(actions, self.action_space_dim)
        preds = (preds * action_masks).sum(dim=-1)
        loss  = (preds - targets.detach()).pow(2) * weights
        prios = loss + 1e-5
        loss  = loss.mean()
        # update priority
        self.buffer.update_priorities(indices, prios.data.numpy())
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

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Sequential
from collections import deque
import numpy as np

'''
Param defs:
env: gym env
lr:  learning rate for gradient descent
gamma: discount rate for future rewards in bellman function
epsilon: exploration rate
epsilon_decay: decay rate for exploration rate after each episode
buffer_size: for experience replay...storing past information on states, actions, rewards, next_states, done
'''

class DoubleDQN(nn.Module):
    def __init__(self, env, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=500000):
        super(DoubleDQN, self).__init__()
        # input output dimensions
        self.state_space_dim = env.observation_space.shape[0]
        self.action_space_dim = env.action_space.n

        # our state-action function estimator

        model = Sequential(
            nn.Linear(self.state_space_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_space_dim)
        )
        self.local_net = model
        self.target_net = model

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.env = env
        self.buffer = deque(maxlen=buffer_size)

        # updates the weights of the model after computing gradients
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=self.lr)

        self.copy = 5000  # Copy the local model weights into the target network every 5000 steps
        self.step = 0

        self.loss_fn = nn.MSELoss()  # defining our loss function to be the MSE loss

    def copy_model(self):
        # Copy local net weights into target net
        self.target_net.load_state_dict(self.local_net.state_dict())

    def insert(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # Get a mini-batch to train the model
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

    def get_action(self, state):
        self.step += 1
        if np.random.uniform() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.local_net(state).data.numpy())

    def train_batch(self, states, actions, rewards, next_states, dones):
        if self.step % self.copy == 0:
            self.copy_model()
        # Bellman equation for updates
        targets = rewards + self.gamma * self.target_net(next_states).max(-1).values * (1.0 - dones)
        preds = self.local_net(states)
        action_masks = F.one_hot(actions, self.action_space_dim)
        preds = (preds * action_masks).sum(dim=-1)
        loss = self.loss_fn(preds, targets.detach())
        self.optimizer.zero_grad()  # zero out the gradients for weights of model
        loss.backward()  # compute the gradient of loss with respect to model parameters
        self.optimizer.step()
        return loss

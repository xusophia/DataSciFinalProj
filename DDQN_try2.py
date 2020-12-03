import gym
import tensorflow as tf
from collections import deque

import random
import numpy as np
import math

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.callbacks import History

# Constants to define:
MAX_EPSILON = 1
MIN_EPSILON = 0.01

GAMMA = 0.95
LAMBDA = 0.0005
TAU = 0.08

BATCH_SIZE = 32
REWARD_STD = 1.0
enviroment = gym.make("LunarLander-v2")

NUM_STATES = 4
NUM_ACTIONS = enviroment.action_space.n

class ExperienceReplay:
    def __init__(self, maxlen = 2000):
        self._buffer = deque(maxlen=maxlen)

    def store(self, state, action, reward, next_state, terminated):
        self._buffer.append((state, action, reward, next_state, terminated))

    def get_batch(self, batch_size):
        if no_samples > len(self._samples):
            return random.sample(self._buffer, len(self._samples))
        else:
            return random.sample(self._buffer, batch_size)

    def get_arrays_from_batch(self, batch):
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([(np.zeros(NUM_STATES) if x[3] is None else x[3])
                                for x in batch])

        return states, actions, rewards, next_states

    @property
    def buffer_size(self):
        return len(self._buffer)


class DDQNAgent:
    def __init__(self, experience_replay, state_size, actions_size, optimizer):

        # Initialize atributes
        self._state_size = state_size
        self._action_size = actions_size
        self._optimizer = optimizer

        self.expirience_replay = experience_replay

        # Initialize discount and exploration rate
        self.epsilon = MAX_EPSILON

        # Build networks
        self.primary_network = self._build_network()
        self.primary_network.compile(loss='mse', optimizer=self._optimizer)

        self.target_network = self._build_network()

     def _build_network(self):
        network = Sequential()
        network.add(Dense(30, activation='relu', kernel_initializer=he_normal()))
        network.add(Dense(30, activation='relu', kernel_initializer=he_normal()))
        network.add(Dense(self._action_size))

        return network

    def align_epsilon(self, step):
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * step)

    def align_target_network(self):
        for t, e in zip(self.target_network.trainable_variables,
                    self.primary_network.trainable_variables): t.assign(t * (1 - TAU) + e * TAU)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self._action_size - 1)
        else:
            q_values = self.primary_network(state.reshape(1, -1))
            return np.argmax(q_values)

    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.store(state, action, reward, next_state, terminated)

    def train(self, batch_size):
        if self.expirience_replay.buffer_size < BATCH_SIZE * 3:
            return 0
        batch = self.expirience_replay.get_batch(batch_size)
        states, actions, rewards, next_states = experience_replay.get_arrays_from_batch(batch)

        # Predict Q(s,a) and Q(s',a') given the batch of states
        q_values_state = self.primary_network(states).numpy()
        q_values_next_state = self.primary_network(next_states).numpy()

        # Copy the q_values_state into the target
        target = q_values_state
        updates = np.zeros(rewards.shape)

        valid_indexes = np.array(next_states).sum(axis=1) != 0
        batch_indexes = np.arange(BATCH_SIZE)

        action = np.argmax(q_values_next_state, axis=1)
        q_next_state_target = self.target_network(next_states)
        updates[valid_indexes] = rewards[valid_indexes] + GAMMA * q_next_state_target.numpy()[batch_indexes[valid_indexes], action[valid_indexes]]

        target[batch_indexes, actions] = updates
        loss = self.primary_network.train_on_batch(states, target)

        # update target network parameters slowly from primary network
        self.align_target_network()

        return loss

def train(self, batch_size):
        if self.expirience_replay.buffer_size < BATCH_SIZE * 3:
            return 0

        batch = self.expirience_replay.get_batch(batch_size)
        states, actions, rewards, next_states = expirience_replay.get_arrays_from_batch(batch)

        # Predict Q(s,a) and Q(s',a') given the batch of states
        q_values_state = self.primary_network(states).numpy()
        q_values_next_state = self.primary_network(next_states).numpy()

        # Initialize target
        target = q_values_state
        updates = np.zeros(rewards.shape)

        valid_indexes = np.array(next_states).sum(axis=1) != 0
        batch_indexes = np.arange(BATCH_SIZE)

        action = np.argmax(q_values_next_state, axis=1)
        q_next_state_target = self.target_network(next_states)
        updates[valid_indexes] = rewards[valid_indexes] + GAMMA *
        q_next_state_target.numpy()[batch_indexes[valid_indexes], action[valid_indexes]]
        target[batch_indexes, actions] = updates
        loss = self.primary_network.train_on_batch(states, target)

        # Slowly update target network parameters from primary network
        self.align_target_network()

        return loss

class AgentTrainer():
    def __init__(self, agent, enviroment):
        self.agent = agent
        self.enviroment = enviroment

    def _take_action(self, action):
        next_state, reward, terminated, _ = self.enviroment.step(action)
        next_state = next_state if not terminated else None
        reward = np.random.normal(1.0, REWARD_STD)
        return next_state, reward, terminated

    def _print_epoch_values(self, episode, total_epoch_reward, average_loss):
        print("**********************************")
        print(f"Episode: {episode} - Reward: {total_epoch_reward} - Average Loss: {average_loss:.3f}")

    def train(self, num_of_episodes = 1000):
        total_timesteps = 0
        for episode in range(0, num_of_episodes):

            # Reset the enviroment
            state = self.enviroment.reset()

            # Initialize variables
            average_loss_per_episode = []
            average_loss = 0
            total_epoch_reward = 0

            terminated = False

            while not terminated:

                # Run Action
                action = agent.act(state)

                # Take action
                next_state, reward, terminated = self._take_action(action)
                agent.store(state, action, reward, next_state, terminated)

                loss = agent.train(BATCH_SIZE)
                average_loss += loss

                state = next_state
                agent.align_epsilon(total_timesteps)
                total_timesteps += 1

                if terminated:
                    average_loss /= total_epoch_reward
                    average_loss_per_episode.append(average_loss)
                    self._print_epoch_values(episode, total_epoch_reward, average_loss)

                # Real Reward is always 1 for Cart-Pole enviroment
                total_epoch_reward +=1


optimizer = Adam()
experience_replay = ExperienceReplay(50000)
agent = DDQNAgent(experience_replay, NUM_STATES, NUM_ACTIONS, optimizer)
agent_trainer = AgentTrainer(agent, enviroment)
agent_trainer.train()

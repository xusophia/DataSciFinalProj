"""

Environment Imports:

"""

import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

# Import Lunar Landing Environment
import gym
env = gym.make('LunarLander-v2')

# CONSTANTS
# AGENT/NETWORK HYPERPARAMETERS
EPSILON_INITIAL = 0.5 # exploration rate
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01
ALPHA = 0.001 # learning rate
GAMMA = 0.99 # discount factor
TAU = 0.1 # target network soft update hyperparameter
EXPERIENCE_REPLAY_BATCH_SIZE = 32
AGENT_MEMORY_LIMIT = 2000
MIN_MEMORY_FOR_EXPERIENCE_REPLAY = 500
OBSERVATION_SPACE_DIMS =  env.observation_space.shape[0]
import torch.optim as optim
device = torch.device('cpu')

def make_dqn():
    model = nn.Sequential(nn.Linear(env.observation_space.shape[0], 512),
                          nn.ReLU(inplace = True),
                          nn.Linear(512, 256),
                          nn.ReLU(inplace = True),
                          nn.Linear(256, 128),
                          nn.ReLU(inplace = True),
                          nn.Linear(128, env.action_space.n))
    loss_function = nn.MSELoss # define MSE as the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    return model, loss_function, optimizer

class DDQNAgent(object):
    def __init__(self):
        self.env = gym.make('LunarLander-v2')
        self.memory = []
        dqn_mod, loss_func, optimizer = make_dqn()
        self.model = dqn_mod
        self.online_network = dqn_mod
        self.target_network = dqn_mod
        self.optimizer = optimizer
        self.loss_fn = loss_func
        self.epsilon = 0.5
        self.has_talked = False

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model(state).data.numpy())

    def experience_replay(self):

        minibatch = random.sample(self.memory, EXPERIENCE_REPLAY_BATCH_SIZE)
        minibatch_new_q_values = []

        for experience in minibatch:
            state, action, reward, next_state, done = experience
            state = self._reshape_state_for_net(state)
            state_tensor = torch.from_numpy(state)
            experience_new_q_values = self.online_network(state_tensor)
            if done:
                q_update = reward
            else:
                next_state = self._reshape_state_for_net(next_state)
                # using online network to SELECT action
                next_state_tensor = torch.from_numpy(next_state)
                online_net_selected_action = np.argmax(self.online_network(next_state_tensor).data.numpy())
                # using target network to EVALUATE action
                target_net_evaluated_q_value = self.target_network(next_state_tensor)
                q_update = reward + GAMMA * target_net_evaluated_q_value
            experience_new_q_values[action%1] = q_update
            minibatch_new_q_values.append(experience_new_q_values)
        minibatch_states = np.array([e[0] for e in minibatch])
        minibatch_new_q_values = np.array(minibatch_new_q_values)
        self.online_network.train(minibatch_states)


    def update_target_network(self):
        q_network_theta = self.online_network.get_weights()
        target_network_theta = self.target_network.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_network_theta,target_network_theta):
            target_weight = target_weight * (1-TAU) + q_weight * TAU
            target_network_theta[counter] = target_weight
            counter += 1
        self.target_network.set_weights(target_network_theta)


    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) <= AGENT_MEMORY_LIMIT:
            experience = (state, action, reward, next_state, done)
            self.memory.append(experience)


    def update_epsilon(self):
        self.epsilon = max(self.epsilon * EPSILON_DECAY, EPSILON_MIN)


    def _reshape_state_for_net(self, state):
        return np.reshape(state,(1, OBSERVATION_SPACE_DIMS))


def test_agent():
    env = gym.make('LunarLander-v2')
    env.seed(1)
    trials = []
    NUMBER_OF_TRIALS = 10
    MAX_TRAINING_EPISODES = 2000
    MAX_STEPS_PER_EPISODE = 200

    for trial_index in range(NUMBER_OF_TRIALS):
        agent = DDQNAgent()
        trial_episode_scores = []

        for episode_index in range(1, MAX_TRAINING_EPISODES+1):
            state = env.reset()
            episode_score = 0

            for _ in range(MAX_STEPS_PER_EPISODE):
                state_tensor = torch.from_numpy(state)
                action = agent.get_action(state_tensor)
                next_state, reward, done, _ = env.step(action)
                episode_score += reward
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if len(agent.memory) > MIN_MEMORY_FOR_EXPERIENCE_REPLAY:
                    agent.experience_replay()
                    agent.update_target_network()
                if done:
                    break

            trial_episode_scores.append(episode_score)
            agent.update_epsilon()
            last_100_avg = np.mean(trial_episode_scores[-100:])
            print ('E %d scored %d, avg %.2f' % (episode_index, episode_score, last_100_avg))
            if len(trial_episode_scores) >= 100 and last_100_avg >= 195.0:
                print ('Trial %d solved in %d episodes!' % (trial_index, (episode_index - 100)))
                break
        trials.append(np.array(trial_episode_scores))
    return np.array(trials)

def plot_trials(trials):
    _, axis = plt.subplots()

    for i, trial in enumerate(trials):
        steps_till_solve = trial.shape[0]-100
        # stop trials at 2000 steps
        if steps_till_solve < 1900:
            bar_color = 'b'
            bar_label = steps_till_solve
        else:
            bar_color = 'r'
            bar_label = 'Stopped at 2000'
        plt.bar(np.arange(i,i+1), steps_till_solve, 0.5, color=bar_color, align='center', alpha=0.5)
        axis.text(i-.25, steps_till_solve + 20, bar_label, color=bar_color)

    plt.ylabel('Episodes Till Solve')
    plt.xlabel('Trial')
    trial_labels = [str(i+1) for i in range(len(trials))]
    plt.xticks(np.arange(len(trials)), trial_labels)
    # remove y axis labels and ticks
    axis.yaxis.set_major_formatter(plt.NullFormatter())
    plt.tick_params(axis='both', left='off')

    plt.title('Double DQN Lunar Landing Trials')
    plt.show()


def plot_individual_trial(trial):
    plt.plot(trial)
    plt.ylabel('Steps in Episode')
    plt.xlabel('Episode')
    plt.title('Double DQN Lunar Landing Steps in Select Trial')
    plt.show()


if __name__ == '__main__':
    trials = test_agent()
    # print 'Saving', file_name
    # np.save('double_dqn_cartpole_trials.npy', trials)
    # trials = np.load('double_dqn_cartpole_trials.npy')
    plot_trials(trials)
    plot_individual_trial(trials[1])

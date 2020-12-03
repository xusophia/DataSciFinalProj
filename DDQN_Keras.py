import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
from matplotlib import pyplot as plt

# AGENT/NETWORK HYPERPARAMETERS
EPSILON_INITIAL = 0.5 # exploration rate
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01
ALPHA = 0.01 # learning rate
GAMMA = 0.99 # discount factor
TAU = 0.05 # target network soft update hyperparameter
EXPERIENCE_REPLAY_BATCH_SIZE = 32
AGENT_MEMORY_LIMIT = 1000
MIN_MEMORY_FOR_EXPERIENCE_REPLAY = 300

# LUNAR LANDING GAME SETTINGS
import gym
env = gym.make('LunarLander-v2')
OBSERVATION_SPACE_DIMS = env.observation_space.shape[0]
ACTION_SPACE = [0,1]

def create_dqn():
    # not actually that deep
    nn = Sequential()
    nn.add(Dense(512, input_dim=OBSERVATION_SPACE_DIMS, activation='relu'))
    nn.add(Dense(512, activation='relu'))
    nn.add(Dense(env.action_space.n, activation='linear'))
    nn.compile(loss='mse', optimizer=Adam(lr=ALPHA))
    return nn


class DoubleDQNAgent(object):


    def __init__(self):
        self.memory = []
        self.online_network = create_dqn()
        self.target_network = create_dqn()
        self.epsilon = EPSILON_INITIAL
        self.has_talked = False


    def act(self, state):
        if self.epsilon > np.random.rand():
            # explore
            return env.action_space.sample()
        else:
            # exploit
            state = self._reshape_state_for_net(state)
            q_values = self.online_network.predict(state)[0]
            return np.argmax(q_values)


    def experience_replay(self):

        minibatch = random.sample(self.memory, EXPERIENCE_REPLAY_BATCH_SIZE)
        minibatch_new_q_values = []

        for experience in minibatch:
            state, action, reward, next_state, done = experience
            state = self._reshape_state_for_net(state)
            experience_new_q_values = self.online_network.predict(state)[0]
            if done:
                q_update = reward
            else:
                next_state = self._reshape_state_for_net(next_state)
                # using online network to SELECT action
                online_net_selected_action = np.argmax(self.online_network.predict(next_state))
                # using target network to EVALUATE action
                target_net_evaluated_q_value = self.target_network.predict(next_state)[0][online_net_selected_action]
                q_update = reward + GAMMA * target_net_evaluated_q_value
            experience_new_q_values[action] = q_update
            minibatch_new_q_values.append(experience_new_q_values)
        minibatch_states = np.array([e[0] for e in minibatch])
        minibatch_new_q_values = np.array(minibatch_new_q_values)
        self.online_network.fit(minibatch_states, minibatch_new_q_values, verbose=False, epochs=1)


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
    NUMBER_OF_TRIALS=10
    MAX_TRAINING_EPISODES = 300
    MAX_STEPS_PER_EPISODE = 200

    for trial_index in range(NUMBER_OF_TRIALS):
        agent = DoubleDQNAgent()
        trial_episode_scores = []

        for episode_index in range(1, MAX_TRAINING_EPISODES+1):
            state = env.reset()
            episode_score = 0

            for _ in range(MAX_STEPS_PER_EPISODE):
                action = agent.act(state)
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

    plt.title('Double DQN LunarLander-v2 Trials')
    plt.show()


def plot_individual_trial(trial):
    plt.plot(trial)
    plt.ylabel('Steps in Episode')
    plt.xlabel('Episode')
    plt.title('Double DQN CLunarLander-v2 Steps in Select Trial')
    plt.show()


if __name__ == '__main__':
    trials = test_agent()
    # print 'Saving', file_name
    # np.save('double_dqn_cartpole_trials.npy', trials)
    # trials = np.load('double_dqn_cartpole_trials.npy')
    plot_trials(trials)
    plot_individual_trial(trials[1])

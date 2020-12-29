from gym import Env, logger
from gym.spaces import Discrete, Tuple, Box
from ray.rllib.env import MultiAgentEnv
from gym.utils import colorize, seeding
import sys
from contextlib import closing
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import pickle

class BertrandCompetitionDiscreteEnv(MultiAgentEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents = 2, c_i = 1, a_minus_c_i = 1, a_0 = 0, mu = 0.25, delta = 0.95, m = 15, xi = 0.1, k = 1, max_steps=200, epochs=10, convergence=5, trainer_choice='DQN'):

        super(BertrandCompetitionDiscreteEnv, self).__init__()
        self.num_agents = num_agents

        # Length of Memory
        self.k = k

        # Marginal Cost
        self.c_i = c_i

        # Number of Discrete Prices
        self.m = m

        # Product Quality Indexes
        self.a = np.array([c_i + a_minus_c_i] * num_agents)

        # Product Quality Index: Outside Good
        self.a_0 = a_0

        # Index of Horizontal Differentiation
        self.mu = mu

        # Nash Equilibrium Price
        # Make sure this tries all possibilities
        price_range = np.arange(0, 2.5, 0.01)
        nash_temp = 0
        for i in price_range:
            p = [i] * num_agents
            first_player_profit = (i - c_i) * self.demand(self.a, p, self.mu, 0)
            new_profit = []
            for j in price_range:
                p[0] = j
                new_profit.append((j - c_i) * self.demand(self.a, p, self.mu, 0))
            if first_player_profit >= np.max(new_profit):
                nash_temp = i
        # Nash Price vs. Marginal Cost
        self.pN = nash_temp
        # self.pN = c_i
        print('Nash Price:', self.pN)

        # Monopoly Equilibrium Price
        monopoly_profit = []
        for i in price_range:
            p = [i] * num_agents
            monopoly_profit.append((i - c_i) * self.demand(self.a, p, self.mu, 0) * num_agents)
        self.pM = price_range[np.argmax(monopoly_profit)]
        print('Monopoly Price:', self.pM)

        # MultiAgentEnv Action Space
        self.action_space = Discrete(m)
        
        # MultiAgentEnv Observation Space
        if k > 0:
            self.numeric_low = np.array([0] * (k * num_agents))
            numeric_high = np.array([m] * (k * num_agents))
            self.observation_space = Box(self.numeric_low, numeric_high, dtype=int)
        else:
            self.numeric_low = np.array([0] * num_agents)
            numeric_high = np.array([m] * num_agents)
            self.observation_space = Box(self.numeric_low, numeric_high, dtype=int)

        self.action_price_space = np.linspace(self.pN - xi * (self.pM - self.pN), self.pM + xi * (self.pM - self.pN), m)
        self.reward_range = (-float('inf'), float('inf'))
        self.current_step = None
        self.max_steps = max_steps
        self.epochs = epochs
        self.convergence = convergence
        self.trainer_choice = trainer_choice
        self.players = [ 'agent_' + str(i) for i in range(num_agents)]
        self.action_history = {}
        self.savefile = 'discrete_' + self.trainer_choice + '_with_' + str(self.num_agents) + '_agents_k_' + str(self.k) + '_for_' + str(self.epochs * self.max_steps) + '_steps'

        for i in range(num_agents):
            if self.players[i] not in self.action_history:
                self.action_history[self.players[i]] = []
                for _ in range(convergence):
                    self.action_history[self.players[i]].append(self.action_space.sample())

        self.reset()

    def demand(self, a, p, mu, agent_idx):
        ''' Demand as a function of product quality indexes, price, and mu. '''
        q = np.exp((a[agent_idx] - p[agent_idx]) / mu) / (np.sum(np.exp((a - p) / mu)) + np.exp(self.a_0 / mu))
        return q

    def step(self, actions_dict):
        ''' MultiAgentEnv Step '''

        actions_idx = np.array(list(actions_dict.values())).flatten()

        with open('./arrays/' + self.savefile + '.pkl', 'ab') as f:
            pickle.dump(actions_idx, f)

        for i in range(actions_idx.size):
            self.action_history[self.players[i]].append(actions_idx[i])

        reward = np.array([0.0] * self.num_agents)

        if self.k > 0:
            obs_players = np.array([self.action_history[self.players[i]][-self.k:] for i in range(self.num_agents)]).flatten()
            observation = dict(zip(self.players, [obs_players for i in range(self.num_agents)]))
        else:
            observation = dict(zip(self.players, [self.numeric_low for _ in range(self.num_agents)]))

        self.prices = self.action_price_space.take(actions_idx)

        for i in range(self.num_agents):
            reward[i] = (self.prices[i] - self.c_i) * self.demand(self.a, self.prices, self.mu, i)

        reward = dict(zip(self.players, reward))
        done = {'__all__': np.all(np.array(self.action_history[self.players[0]][-self.convergence:]) == self.action_history[self.players[0]][-1])
                                   or self.current_step == self.max_steps}
        info = dict(zip(self.players, [{}]*self.num_agents))

        self.current_step += 1

        return observation, reward, done, info

    def deviate(self, direction='down'):
        deviate_actions_dict = {}

        if direction == 'down':
            # First player deviates to lowest price
            deviate_actions_dict[self.players[0]] = 0
        elif direction == 'up':
            # First player deviates to highest price
            deviate_actions_dict[self.players[0]] = self.m - 1

        for agent in range(1, self.num_agents):
            # All other player remain at previous price (large assumption)
            deviate_actions_dict[self.players[agent]] = self.action_history[self.players[agent]][-1]

        observation, _, _, _ = self.step(deviate_actions_dict)

        return observation

    def reset(self):
        self.current_step = 0

        # Reset to random action
        random_action = np.random.randint(self.m, size=self.num_agents)

        for i in range(random_action.size):
            self.action_history[self.players[i]].append(random_action[i])

        if self.k > 0:
            obs_players = np.array([self.action_history[self.players[i]][-self.k:] for i in range(self.num_agents)]).flatten()
            observation = dict(zip(self.players, [obs_players for i in range(self.num_agents)]))
        else:
            observation = dict(zip(self.players, [self.numeric_low for _ in range(self.num_agents)]))
            
        return observation

    def plot(self):
        '''Plot action history.'''
        n = len(self.action_history[self.players[0]])
        x = np.arange(n)
        for player in self.players:
            plt.plot(x, self.action_price_space.take(self.action_history[player]), alpha=0.75, label=player)
        plt.plot(x, np.repeat(self.pM, n), 'r--', label='Monopoly')
        plt.plot(x, np.repeat(self.pN, n), 'b--', label='Nash')
        plt.xlabel('Steps')
        plt.ylabel('Price')
        plt.title(self.trainer_choice + ' with ' + str(self.num_agents) + ' agents and k=' + str(self.k) + ' for ' + str(self.epochs * self.max_steps) + ' Steps')
        plt.legend(loc='upper left')
        plt.savefig('./figures/' + self.savefile)
        plt.clf()

    def plot_last(self, last_n=1000, title_str = ''):
        x = np.arange(last_n)
        for player in self.players:
            plt.plot(x, self.action_price_space.take(self.action_history[player][-last_n:]), alpha=0.75, label=player)
        plt.plot(x, np.repeat(self.pM, last_n), 'r--', label='Monopoly')
        plt.plot(x, np.repeat(self.pN, last_n), 'b--', label='Nash')
        plt.xlabel('Steps')
        plt.ylabel('Price')
        plt.title(self.trainer_choice + ' with ' + str(self.num_agents) + ' agents and k=' + str(self.k) + ' for ' + str(self.epochs * self.max_steps) + ' Steps, Last Steps' + str(last_n))
        plt.legend()
        plt.savefig('./figures/' + self.savefile + title_str + '_last_steps_' + str(last_n))
        plt.clf()

    def render(self, mode='human'):
        raise NotImplementedError
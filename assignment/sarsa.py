import numpy as np
from logging import getLogger


class SarsaEpisode(object):
    _logger = getLogger('assignment.sarsa.episode')

    def __init__(self, environment, parametrised_qs, policy, max_steps):
        self._environment = environment
        self._parametrised_qs = parametrised_qs
        self._policy = policy
        self._max_steps = max_steps
        self._step = 0

    def run_episode(self):
        state = self._environment.state
        qs = self._parametrised_qs.compute(state)
        action = self._policy(qs)
        while not (self._environment.terminated or
                   self._step > self._max_steps):
            reward = self._environment.perform_action(action)
            new_state = self._environment.state

            new_qs = self._parametrised_qs.compute(new_state)
            new_action = self._policy(new_qs)

            self._parametrised_qs.update(state, action, reward, new_state,
                                         new_action,
                                         cached={'qs': qs, 'new_qs': new_qs})

            state = new_state
            action = new_action

            self._logger.debug(('Completed 1 step; environment = {}, ' +
                                'parametrised_qs = {}')
                               .format(self._environment,
                                       self._parametrised_qs))


class SarsaRun(object):
    _logger = getLogger('assignment.sarsa.run')

    def __init__(self, num_episodes, episode_factory):
        self._num_episodes = num_episodes
        self._episode_factory = episode_factory

    def run(self):
        for ii in range(self._num_episodes):
            episode = self._episode_factory()
            episode.run_episode()
            self._logger.debug(('Completed {} episodes out of {}; ' +
                                'episode_factory = {}')
                               .format(ii, self._num_episodes,
                                       self._episode_factory))


class BasicQs(object):
    _logger = getLogger('assignment.sarsa.qs.basicqs')

    def __init__(self, initial_value, num_states, num_actions, learning_rate,
                 discount_rate):
        self._qs = np.full((num_states, num_actions), initial_value)
        self._learning_rate = learning_rate
        self._discount_rate = discount_rate

    def compute(self, state):
        return self._qs[state, :]

    def update(self, state, action, reward, new_state, new_action,
               cached=None):
        dq = self._learning_rate * (reward +
                                    (self._discount_rate *
                                     self._qs[new_state, new_action]) -
                                    self._qs[state, action])
        selector = np.zeros(self._qs.shape)
        selector[state, action] = 1
        self._qs += selector * dq

        self._logger.debug('Updated Q-values; dq = {}'.format(dq))


class EpsilonGreedy(object):
    _logger = getLogger('assignment.sarsa.policy.epsilongreedy')

    def __init__(self, epsilon, num_actions):
        self._epsilon = epsilon
        self._num_actions = num_actions

    def __call__(self, qs):
        if np.random.rand() < self._epsilon:
            action = np.random.randint(self._num_actions)
            self._logger.debug('Choosing randomly: {}'.format(action))
        else:
            action = np.argmax(qs)
            self._logger.debug('Choosing best: {}'.format(action))
        return action

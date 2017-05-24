import numpy as np
from logging import getLogger


class Policy(object):
    """Abstract class for a policy."""

    def __init__(self, num_actions):
        self._num_actions = num_actions

    def choose_action(self, qs, t):
        raise NotImplementedError()


class EpsilonGreedy(Policy):
    """Epsilon-greedy policy

    Chooses the best action with probablity (1-epsilon), otherwise chooses
    randomly."""
    _logger = getLogger('assignment.sarsa.policy.epsilongreedy')

    def __init__(self, num_actions, epsilon):
        super().__init__(num_actions)
        self._epsilon = epsilon

    def choose_action(self, qs, _):
        if np.random.rand() < self._epsilon or np.allclose(qs[0], qs):
            action = np.random.randint(self._num_actions)
            self._logger.debug('Choosing randomly: {}'.format(action))
        else:
            action = np.argmax(qs)
            self._logger.debug('Choosing best: {}'.format(action))
        return action


class EpsilonGreedyDecay(Policy):
    """Epsilon-greedy policy with decay

    Same as epsilon-greedy, but the value of epsilon decays with the episode
    number."""
    _logger = getLogger('assignment.sarsa.policy.epsilongreedy')

    def __init__(self, num_actions, epsilon):
        super().__init__(num_actions)
        self._epsilon = epsilon

    def choose_action(self, qs, episode_number):
        if (np.random.rand() < (self._epsilon / (episode_number + 1)) or
                np.allclose(qs[0], qs)):
            action = np.random.randint(self._num_actions)
            self._logger.debug('Choosing randomly: {}'.format(action))
        else:
            action = np.argmax(qs)
            self._logger.debug('Choosing best: {}'.format(action))
        return action


class SoftMaxDecay(Policy):
    def __init__(self, num_actions, tau):
        super().__init__(num_actions)
        self._tau = tau

    def choose_action(self, qs, t):
        boltzmann_numerator = np.exp((t * qs) / self._tau)
        probabilities = boltzmann_numerator / np.sum(boltzmann_numerator)

        return np.random.choice(self._num_actions, p=probabilities)

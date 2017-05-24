import numpy as np
from logging import getLogger


class QModel(object):
    """Abstract class of Q-value model."""

    def __init__(self, num_states, num_actions):
        self._num_states = num_states
        self._num_actions = num_actions

    def compute(self, state):
        raise NotImplementedError()

    def update(self, state, action, reward, new_state, new_action,
               episode_number, cached=None):
        raise NotImplementedError()

    @property
    def parameters(self):
        raise NotImplementedError()

    @property
    def qs(self):
        raise NotImplementedError()


class BasicQs(QModel):
    """Basic Q-value model

    Keeps Q-values in a table and updates them towards the target."""
    _logger = getLogger('assignment.sarsa.qs.basicqs')

    def __init__(self, initial_value, num_states, num_actions, learning_rate,
                 discount_rate):
        super().__init__(num_states, num_actions)
        self._qs = np.full((num_states, num_actions), float(initial_value))
        self._learning_rate = learning_rate
        self._discount_rate = discount_rate

    def compute(self, state):
        return self._qs[state, :]

    def update(self, state, action, reward, new_state, new_action,
               episode_number, cached=None):
        dq = self._learning_rate * (reward +
                                    (self._discount_rate *
                                     cached['new_qs'][new_action]) -
                                    cached['qs'][action])
        selector = np.zeros(self._qs.shape)
        selector[state, action] = 1
        self._qs += selector * dq

        self._logger.debug('Updated Q-values; dq = {}'.format(dq))

    @property
    def parameters(self):
        return self._qs

    @property
    def qs(self):
        return self._qs


class BasicQsEligibilityTrace(BasicQs):
    """Basic Q-value model with an eligibility trace"""

    _logger = getLogger('assignment.sarsa.qs.basicqs_eligibility')

    def __init__(self, initial_value, num_states, num_actions, learning_rate,
                 discount_rate, trace_decay_rate):
        super().__init__(initial_value, num_states, num_actions, learning_rate,
                         discount_rate)
        self._trace_decay_rate = trace_decay_rate
        self._trace = np.zeros((num_states, num_actions))
        self._current_episode_number = None

    def update(self, state, action, reward, new_state, new_action,
               episode_number, cached=None):
        # Reset the eligibility trace at the start of each episode
        if episode_number != self._current_episode_number:
            self._trace = np.zeros(self._trace.shape)
            self._current_episode_number = episode_number

        dq = self._learning_rate * (reward +
                                    (self._discount_rate *
                                     self._qs[new_state, new_action]) -
                                    self._qs[state, action])
        self._trace[state, action] = self._trace[state, action] + 1

        self._qs += self._trace * dq
        self._trace *= self._trace_decay_rate

        self._logger.debug('Updated Q-values; dq = {}'.format(dq))


class NeuralQs(QModel):
    """Model using a neural network to generate Q-values"""

    _logger = getLogger('assignment.sarsa.qs.neural')

    def __init__(self, num_states, num_actions, learning_rate, discount_rate):
        super().__init__(num_states, num_actions)
        self._weights = np.random.rand(num_states, num_actions)
        self._learning_rate = learning_rate
        self._discount_rate = discount_rate

    def compute(self, state):
        actions_inputs = self._weights[state, :]
        qvalues = 1 / (1 + np.exp(-actions_inputs))
        return qvalues

    def update(self, state, action, reward, new_state, new_action,
               episode_number, cached=None):
        dw = self._learning_rate * (reward +
                                    (self._discount_rate *
                                     cached['new_qs'][new_action]) -
                                    cached['qs'][action])
        selector = np.zeros(self._weights.shape)
        selector[state, action] = 1
        self._weights += selector * dw

        self._logger.debug('Updated weights; dw = {}'.format(dw))

    @property
    def qs(self):
        return self._weights


class NeuralQsEligibility(NeuralQs):
    """Neural network Q-value model with an eligibility trace"""

    _logger = getLogger('assignment.sarsa.qs.neural')

    def __init__(self, num_states, num_actions, learning_rate, discount_rate,
                 trace_decay_rate):
        super().__init__(num_states, num_actions, learning_rate, discount_rate)
        self._trace_decay_rate = trace_decay_rate
        self._trace = np.zeros((num_states, num_actions))
        self._current_episode_number = None

    def update(self, state, action, reward, new_state, new_action,
               episode_number, cached=None):
        # Reset the eligibility trace at the start of each episode
        if episode_number != self._current_episode_number:
            self._trace = np.zeros(self._trace.shape)
            self._current_episode_number = episode_number
        dw = self._learning_rate * (reward +
                                    (self._discount_rate *
                                     cached['new_qs'][new_action]) -
                                    cached['qs'][action])
        selector = np.zeros(self._weights.shape)
        selector[state, action] = 1
        self._weights += selector * dw
        self._trace[state, action] = self._trace[state, action] + 1

        self._weights += self._trace * dw
        self._trace *= self._trace_decay_rate

        self._logger.debug('Updated weights; dw = {}'.format(dw))

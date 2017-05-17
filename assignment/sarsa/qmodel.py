import numpy as np
from logging import getLogger


class QModel(object):
    def compute(self, state):
        raise NotImplementedError()

    def update(self, state, action, reward, new_state, new_action,
               cached=None):
        raise NotImplementedError()

    @property
    def parameters(self):
        raise NotImplementedError()

    @property
    def qs(self):
        raise NotADirectoryError()


class BasicQs(QModel):
    _logger = getLogger('assignment.sarsa.qs.basicqs')

    def __init__(self, initial_value, num_states, num_actions, learning_rate,
                 discount_rate):
        self._qs = np.full((num_states, num_actions), float(initial_value))
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

    @property
    def parameters(self):
        return self._qs

    @property
    def qs(self):
        return self._qs

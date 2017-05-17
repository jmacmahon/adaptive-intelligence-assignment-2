# TODO profile and investigate potential numpy speedup

from random import randint
from logging import getLogger


class ImageMonkey(object):
    _logger = getLogger('assignment.environment.imagemonkey')
    # States: 0 = neutral, 1 = pressed red, 2 = pressed green

    def __init__(self):
        self.reset()

    def reset(self):
        self._state = 0
        self._buttons_pressed = 0

    @property
    def num_states(self):
        return 3

    @property
    def num_actions(self):
        return 2

    @property
    def state(self):
        return self._state

    @property
    def terminated(self):
        return self._buttons_pressed >= 2

    def perform_action(self, action_index):
        old_state = self._state
        if self.terminated:
            raise RuntimeError("Already terminated")
        if action_index == 0:
            self._state = 1
            reward = 0
        elif action_index == 1:
            if self._state == 1:
                reward = 1
            else:
                reward = 0
            self._state = 2
        self._buttons_pressed += 1
        self._logger.debug(('Got action = {} in state = {}: reward = {}, ' +
                            'new state = {}')
                           .format(action_index, old_state, reward,
                                   self._state))
        return reward


class HomingRobot(object):
    _logger = getLogger('assignment.environment.homingrobot')

    # Down, up, right, left
    _actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def __init__(self, width, height, home_coords):
        self._width = width
        self._height = height
        self._home_coords = home_coords
        self.reset()

    def reset(self):
        self._x = randint(0, self._width - 1)
        self._y = randint(0, self._height - 1)

    @property
    def num_states(self):
        return self._width * self._height

    @property
    def num_actions(self):
        return len(self._actions)

    @property
    def state(self):
        return self._x + self._height * self._y

    @property
    def coords(self):
        return self._x, self._y

    @property
    def terminated(self):
        return (self._x, self._y) == self._home_coords

    def perform_action(self, action_index):
        if self.terminated:
            raise RuntimeError("Already terminated")
        action = self._actions[action_index]
        new_x = self._x + action[0]
        new_y = self._y + action[1]

        if new_x < 0 or new_x >= self._width:
            new_x = self._x
        if new_y < 0 or new_y >= self._height:
            new_y = self._y

        self._x, self._y = new_x, new_y

        if self.terminated:
            reward = 10
        else:
            reward = 0

        self._logger.debug(('Moved; action_index = {}, new x = {}, new y ' +
                            '= {}, reward = {}')
                      .format(action_index, new_x, new_y, reward))
        return reward

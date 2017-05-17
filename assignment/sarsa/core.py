import numpy as np
from logging import getLogger


class SarsaEpisode(object):
    _logger = getLogger('assignment.sarsa.episode')

    def __init__(self, environment, parametrised_qs, policy, max_steps):
        self._environment = environment
        self._parametrised_qs = parametrised_qs
        self._policy = policy
        self._max_steps = max_steps

    def run_episode(self):
        self._environment.reset()
        state = self._environment.state
        qs = self._parametrised_qs.compute(state)
        step = 0
        action = self._policy.choose_action(qs, step)
        while not (self._environment.terminated or step > self._max_steps):
            step += 1
            reward = self._environment.perform_action(action)
            new_state = self._environment.state

            new_qs = self._parametrised_qs.compute(new_state)
            new_action = self._policy.choose_action(new_qs, step)

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

    def __init__(self, num_episodes, episode):
        self._num_episodes = num_episodes
        self._episode = episode

    def run(self):
        for ii in range(self._num_episodes):
            self._episode.run_episode()
            self._logger.info('Completed {} episodes out of {}; episode = {}'
                               .format(ii + 1, self._num_episodes,
                                       self._episode))

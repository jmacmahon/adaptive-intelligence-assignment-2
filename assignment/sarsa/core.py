import numpy as np
from logging import getLogger


class SarsaEpisode(object):
    _logger = getLogger('assignment.sarsa.episode')

    def __init__(self, environment, parametrised_qs, policy, max_steps):
        self._environment = environment
        self._parametrised_qs = parametrised_qs
        self._policy = policy
        self._max_steps = max_steps

    def run_episode(self, episode_number):
        self._environment.reset()
        state = self._environment.state
        qs = self._parametrised_qs.compute(state)
        step = 0
        action = self._policy.choose_action(qs, episode_number)
        total_reward = 0
        while not (self._environment.terminated or step > self._max_steps):
            step += 1
            reward = self._environment.perform_action(action)
            total_reward += reward
            new_state = self._environment.state

            new_qs = self._parametrised_qs.compute(new_state)
            new_action = self._policy.choose_action(new_qs, episode_number)

            self._parametrised_qs.update(state, action, reward, new_state,
                                         new_action, episode_number,
                                         cached={'qs': qs, 'new_qs': new_qs})

            state = new_state
            action = new_action
            qs = new_qs

            self._logger.debug(('Completed 1 step; environment = {}, ' +
                                'parametrised_qs = {}')
                               .format(self._environment,
                                       self._parametrised_qs))
        return step, total_reward


class SarsaRun(object):
    _logger = getLogger('assignment.sarsa.run')

    def __init__(self, num_episodes, episode):
        self._num_episodes = num_episodes
        self._episode = episode

    def run(self):
        steps = np.empty(self._num_episodes)
        rewards = np.empty(self._num_episodes)
        for ii in range(self._num_episodes):
            num_steps, total_reward = self._episode.run_episode(
                episode_number=ii)
            steps[ii] = num_steps
            rewards[ii] = total_reward
            self._logger.debug(('Completed {} episodes out of {}; episode ' +
                               '= {}, num_steps = {}, total_reward = {}')
                               .format(ii + 1, self._num_episodes,
                                       self._episode, num_steps, total_reward))
        return steps, rewards

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
                                         new_action,
                                         cached={'qs': qs, 'new_qs': new_qs})

            state = new_state
            action = new_action

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


class SarsaMultipleRuns(object):
    _logger = getLogger('assignment.sarsa.multipleruns')

    def __init__(self, num_runs, num_episodes, max_episode_step, environment,
                 policy_partial, parametrised_qs_partial):
        self._num_runs = num_runs
        self._num_episodes = num_episodes
        self._max_episode_step = max_episode_step
        self._environment = environment
        self._policy_partial = policy_partial
        self._parametrised_qs_partial = parametrised_qs_partial

    def run(self):
        num_states = self._environment.num_states
        num_actions = self._environment.num_actions

        step_curves = np.empty((self._num_runs, self._num_episodes))
        reward_curves = np.empty((self._num_runs, self._num_episodes))
        for run_index in range(self._num_runs):
            policy = self._policy_partial(num_actions=num_actions)
            parametrised_qs = self._parametrised_qs_partial(
                num_states=num_states,
                num_actions=num_actions)
            episode = SarsaEpisode(self._environment, parametrised_qs, policy,
                                   self._max_episode_step)
            run = SarsaRun(self._num_episodes, episode)
            step_curve, reward_curve = run.run()
            step_curves[run_index, :] = step_curve
            reward_curves[run_index, :] = reward_curve
            self._logger.info('Completed run {} of {}'
                              .format(run_index + 1, self._num_runs))
        return step_curves, reward_curves

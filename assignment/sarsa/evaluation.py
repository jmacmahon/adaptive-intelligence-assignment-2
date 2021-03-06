import numpy as np
from logging import getLogger
from multiprocessing import Pool

from .core import SarsaRun, SarsaEpisode


class SarsaMultipleRuns(object):
    """Carry out multiple independent runs of the Sarsa algorithm

    Performs n runs in parallel (with multiprocessing) and returns n learning
    curves.
    """
    _logger = getLogger('assignment.sarsa.multipleruns')

    def __init__(self, num_runs, num_episodes, max_episode_step, environment,
                 policy_partial, parametrised_qs_partial):
        self._num_runs = num_runs
        self._num_episodes = num_episodes
        self._max_episode_step = max_episode_step
        self._environment = environment
        self._policy_partial = policy_partial
        self._parametrised_qs_partial = parametrised_qs_partial

    def build_run(self):
        num_states = self._environment.num_states
        num_actions = self._environment.num_actions

        policy = self._policy_partial(num_actions=num_actions)
        parametrised_qs = self._parametrised_qs_partial(
            num_states=num_states,
            num_actions=num_actions)
        episode = SarsaEpisode(self._environment, parametrised_qs, policy,
                               self._max_episode_step)
        run = SarsaRun(self._num_episodes, episode)
        return run

    def _run_one(self, run_index):
        run = self.build_run()
        step_curve, reward_curve = run.run()
        self._logger.info('Completed run {} of {}'
                          .format(run_index + 1, self._num_runs))
        return step_curve, reward_curve

    def run(self, processes=6):
        pool = Pool(processes)
        curves = pool.map(self._run_one, range(self._num_runs))
        pool.close()
        pool.join()
        step_curves, reward_curves = zip(*curves)
        return np.array(step_curves), np.array(reward_curves)

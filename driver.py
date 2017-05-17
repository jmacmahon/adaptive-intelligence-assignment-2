import numpy as np
import coloredlogs
import cProfile as profile
import pstats
import matplotlib.pyplot as plt
from functools import partial
from itertools import product
from logging import getLogger
from pickle import load, dump

from assignment.environment import *
from assignment.sarsa.core import *
from assignment.sarsa.policy import *
from assignment.sarsa.qmodel import *
from assignment.sarsa.evaluation import *
from assignment.display import *

NUM_PROCESSES = 6

coloredlogs.install(level='WARN')
coloredlogs.install(level='INFO', logger=getLogger('assignment.driver'))

e_greedy_decay_policy_partial = partial(EpsilonGreedyDecay, epsilon=0.5)
e_greedy_policy_partial = partial(EpsilonGreedyDecay, epsilon=0.1)
basic_qs_partial = partial(BasicQs, initial_value=0,
                           learning_rate=0.8, discount_rate=0.9)
basic_qs_eligibility_partial = partial(BasicQsEligibilityTrace,
                                       initial_value=0, learning_rate=0.8,
                                       discount_rate=0.9, trace_decay_rate=0.5)
nn_qs_partial = partial(NeuralQs, learning_rate=0.8, discount_rate=0.9)
nn_qs_eligibility_partial = partial(NeuralQsEligibility, learning_rate=2,
                                    discount_rate=0.6, trace_decay_rate=0.5)


def question3(num_runs=20, num_episodes=200, max_episode_step=20,
              hr_environment=None):
    """Optimise learning rate, discount rate and epsilon"""

    logger = getLogger('assignment.driver.q3')

    params = {
        'learning_rate': np.arange(0.2, 4, 0.4),
        'discount_rate': np.array([0.3, 0.5, 0.6, 0.7, 0.8, 0.9])
        # 'epsilon': np.arange(0, 0.5, 0.1),
        # 'trace_decay_rate': np.array([0.5]), #np.arange(0, 1, 0.2),
    }

    # TODO graph epsilon and trace_decay_rate

    if hr_environment is None:
        hr_environment = HomingRobot(10, 10, (5, 5), 10, 0)

    detailed_results = []
    total_combinations = len(list(product(*params.values())))
    i = 0
    results = np.empty((total_combinations, len(params) + 1))
    for values in product(*params.values()):
        kwargs = dict(zip(params.keys(), values))
        policy_partial = partial(EpsilonGreedy,
                                 epsilon=0.1)
        qs_partial = partial(NeuralQsEligibility,
                             learning_rate=kwargs['learning_rate'],
                             discount_rate=kwargs['discount_rate'],
                             trace_decay_rate=0.5)
        runs = SarsaMultipleRuns(num_runs, num_episodes, max_episode_step,
                                 hr_environment, policy_partial, qs_partial)
        step_curve, _ = runs.run()

        # Weighted average steps
        avg_curve = np.mean(step_curve, axis=0)
        evaluation_metric = np.sum(np.linspace(0, 1, num_episodes) * avg_curve)
        # evaluation_metric = np.sum(avg_curve[-100:])

        results[i, :] = values + (evaluation_metric,)
        detailed_results.append((kwargs, step_curve))
        i += 1
        logger.info('Evaluated parameter combination {} of {}; values = {}'
                    .format(i, total_combinations, values))
    return detailed_results, results


def question3_load_pickle(commit='362a88b'):
    with open('q3_results_{}.pickle'.format(commit), 'rb') as f:
        return load(f)


def image_monkey():
    im = ImageMonkey()
    sarsa_runs = SarsaMultipleRuns(100, 100, 20, im,
                                   e_greedy_decay_policy_partial,
                                   basic_qs_eligibility_partial)
    return sarsa_runs


def homing_robot():
    hr = HomingRobot(10, 10, (5, 5), 10, 0)
    sarsa_runs = SarsaMultipleRuns(100, 200, 30, hr,
                                   e_greedy_decay_policy_partial,
                                   nn_qs_eligibility_partial)
    return sarsa_runs


def plot_runs(runs):
    step_curves, reward_curves = runs.run(NUM_PROCESSES)

    mean_step_curve = np.mean(step_curves, axis=0)
    errorbars_step_curve = (np.std(step_curves, axis=0) /
                            np.sqrt(step_curves.shape[0]))

    mean_reward_curve = np.mean(reward_curves, axis=0)
    errorbars_reward_curve = (np.std(reward_curves, axis=0) /
                              np.sqrt(reward_curves.shape[0]))

    fig, (step_axes, reward_axes) = plt.subplots(2, 1)
    step_axes.errorbar(x=np.arange(step_curves.shape[1]),
                       y=mean_step_curve,
                       yerr=errorbars_step_curve)
    reward_axes.errorbar(x=np.arange(reward_curves.shape[1]),
                         y=mean_reward_curve,
                         yerr=errorbars_reward_curve)
    plt.show()

# if __name__ == '__main__':
#     runs = homing_robot()
#     curves = runs.run()
#     print(curves)

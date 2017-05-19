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

hr_environment = HomingRobot(10, 10, (5, 5), 10, 0)


# from https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def question1_single_curves(num_episodes=1000, max_episode_step=20,
                            epsilon=0.001, avg_over=50):
    egreedy_partial = partial(EpsilonGreedy, epsilon=epsilon)
    basic_qs_partial = partial(BasicQs, initial_value=0, learning_rate=0.1,
                               discount_rate=0.1)
    runs = SarsaMultipleRuns(100, num_episodes, max_episode_step,
                             hr_environment, egreedy_partial, basic_qs_partial)
    run1 = runs.build_run()
    run2 = runs.build_run()
    step_curve_1, _ = run1.run()
    step_curve_2, _ = run2.run()

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(moving_average(step_curve_1, n=avg_over))
    ax1.set_xlabel('Episode number')
    ax1.set_ylabel('Steps taken to goal')
    ax2.plot(moving_average(step_curve_2, n=avg_over))
    ax2.set_xlabel('Episode number')
    ax2.set_ylabel('Steps taken to goal')
    plt.show()


def question1_avg_curve(*args, epsilon=0.001, **kwargs):
    egreedy_partial = partial(EpsilonGreedy, epsilon=epsilon)
    basic_qs_partial = partial(BasicQs, initial_value=0, learning_rate=0.1,
                               discount_rate=0.1)
    models = [
        {'policy': egreedy_partial,
         'qs': basic_qs_partial,
         'label': 'Averaged learning curve'}
    ]

    compare(models)


def nn_basic_eligibility_comparison(*args, epsilon=0.2, **kwargs):
    egreedy_partial = partial(EpsilonGreedy, epsilon=epsilon)
    learning_rate = 0.8
    discount_rate = 0.6
    trace_decay_rate = 0.5

    models = [
       {'policy': egreedy_partial,
        'qs': partial(BasicQs, initial_value=0, learning_rate=learning_rate,
                      discount_rate=discount_rate),
        'label': 'Basic model without λ'},
        {'policy': egreedy_partial,
         'qs': partial(BasicQsEligibilityTrace, initial_value=0,
                       learning_rate=learning_rate,
                       discount_rate=discount_rate,
                       trace_decay_rate=trace_decay_rate),
         'label': 'Basic model with λ = {}'.format(trace_decay_rate)},
        {'policy': egreedy_partial,
         'qs': partial(NeuralQs, learning_rate=learning_rate,
                       discount_rate=discount_rate),
         'label': 'ANN model without λ'},
        {'policy': egreedy_partial,
         'qs': partial(NeuralQsEligibility, learning_rate=learning_rate,
                       discount_rate=discount_rate,
                       trace_decay_rate=trace_decay_rate),
         'label': 'ANN model with λ = {}'.format(trace_decay_rate)}
    ]
    return compare(models, *args, **kwargs)


def compare(models, num_runs=100, num_episodes=1000, max_episode_step=20,
            graph_step=10):
    logger = getLogger('assignment.driver.compare')

    xs = np.arange(num_episodes)[::graph_step]
    fig, axes = plt.subplots(1, 1)
    ii = 0
    for model in models:
        ii += 1
        runs = SarsaMultipleRuns(num_runs, num_episodes, max_episode_step,
                                 hr_environment, model['policy'], model['qs'])
        step_curves, _ = runs.run()
        step_curves = step_curves[:, ::graph_step]

        mean_step_curve = np.mean(step_curves, axis=0)
        errorbars_step_curve = (np.std(step_curves, axis=0) /
                                np.sqrt(step_curves.shape[0]))

        axes.errorbar(x=xs,
                      y=mean_step_curve,
                      yerr=errorbars_step_curve,
                      label=model['label'])
        axes.set_xlabel('Episode number')
        axes.set_ylabel('Steps taken to goal')

        logger.info('Done model {} of {}: {}'
                    .format(ii, len(models), model['label']))
    axes.legend()
    plt.show()


def question3_lr_dr(num_runs=20, num_episodes=200, max_episode_step=20,
                    epsilon=0.1, trace_decay_rate=0.5):
    """Optimise learning rate, discount rate"""

    logger = getLogger('assignment.driver.q3.lr_dr')

    params = {
        'learning_rate': np.arange(0.2, 4, 0.4),
        'discount_rate': np.array([0.3, 0.5, 0.6, 0.7, 0.8, 0.9])
    }

    detailed_results = []
    total_combinations = len(list(product(*params.values())))
    i = 0
    results = np.empty((total_combinations, len(params) + 1))
    for values in product(*params.values()):
        kwargs = dict(zip(params.keys(), values))
        policy_partial = partial(EpsilonGreedy, epsilon=epsilon)
        qs_partial = partial(NeuralQsEligibility,
                             learning_rate=kwargs['learning_rate'],
                             discount_rate=kwargs['discount_rate'],
                             trace_decay_rate=trace_decay_rate)
        runs = SarsaMultipleRuns(num_runs, num_episodes, max_episode_step,
                                 hr_environment, policy_partial, qs_partial)
        step_curves, _ = runs.run()

        # Weighted average steps
        avg_curve = np.mean(step_curves, axis=0)
        evaluation_metric = np.sum(np.linspace(0, 1, num_episodes) * avg_curve)
        # evaluation_metric = np.sum(avg_curve[-100:])

        results[i, :] = values + (evaluation_metric,)
        detailed_results.append((kwargs, step_curves))
        i += 1
        logger.info('Evaluated parameter combination {} of {}; values = {}'
                    .format(i, total_combinations, values))
    get_3d_tunings_figures(results, labels=list(params.keys()))
    plt.show()


def question3_epsilon(*args, **kwargs):
    qs_partial = partial(NeuralQsEligibility, learning_rate=0.8,
                         discount_rate=0.6, trace_decay_rate=0.5)

    epsilons = [(0, False),
                (0.05, False),
                (0.1, False),
                (0.5, False),
                (1, False),
                (1, True),
                (10, True),
                (100, True)]
    models = [
        {'policy': partial((EpsilonGreedyDecay if decay_flag
                            else EpsilonGreedy), epsilon=epsilon),
         'qs': qs_partial,
         'label': ('ε = {}, with decay' if decay_flag
                   else 'ε = {}, no decay')
                   .format(epsilon)}
        for (epsilon, decay_flag) in epsilons
    ]
    compare(models, *args, **kwargs)


def question3_tdr(*args, **kwargs):
    policy_partial = partial(EpsilonGreedyDecay, epsilon=0.1)

    trace_decay_rates = [0.0, 0.2, 0.4, 0.5, 0.7, 0.9]
    models = [
        {'policy': policy_partial,
         'qs': partial(NeuralQsEligibility, learning_rate=0.8,
                       discount_rate=0.6,
                       trace_decay_rate=trace_decay_rate),
         'label': 'λ = {:.1}'.format(trace_decay_rate)}
        for trace_decay_rate in trace_decay_rates
    ]
    compare(models, *args, **kwargs)

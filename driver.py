import numpy as np
import coloredlogs
import cProfile as profile
import pstats
import matplotlib.pyplot as plt
from functools import partial

from assignment.environment import *
from assignment.sarsa.core import *
from assignment.sarsa.policy import *
from assignment.sarsa.qmodel import *

coloredlogs.install(level='INFO')

e_greedy_decay_policy_partial = partial(EpsilonGreedyDecay, epsilon=0.5)
e_greedy_policy_partial = partial(EpsilonGreedyDecay, epsilon=0.1)
basic_qs_partial = partial(BasicQs, initial_value=0,
                           learning_rate=0.8, discount_rate=0.9)

def image_monkey():
    im = ImageMonkey()
    sarsa_runs = SarsaMultipleRuns(100, 400, 20, im, e_greedy_policy_partial,
                                   basic_qs_partial)
    return sarsa_runs

def homing_robot():
    hr = HomingRobot(10, 10, (5, 5))
    sarsa_runs = SarsaMultipleRuns(100, 200, 30, hr,
                                   e_greedy_decay_policy_partial,
                                   basic_qs_partial)
    return sarsa_runs

def moving_average(x, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(x, window, 'same')

def plot_runs(runs):
    step_curves, reward_curves = runs.run()
    mean_step_curve = np.mean(step_curves, axis=0)
    mean_reward_curve = np.mean(reward_curves, axis=0)
    fig, (step_axes, reward_axes) = plt.subplots(2, 1)
    step_axes.plot(mean_step_curve)
    reward_axes.plot(mean_reward_curve)
    plt.show()

# if __name__ == '__main__':
#     runs = homing_robot()
#     curves = runs.run()
#     print(curves)

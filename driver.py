import numpy as np
import coloredlogs
import cProfile as profile
import pstats

from assignment.environment import *
from assignment.sarsa.core import *
from assignment.sarsa.policy import *
from assignment.sarsa.qmodel import *

coloredlogs.install(level='INFO')

def image_monkey():
    im = ImageMonkey()
    epsilon_greedy_policy = EpsilonGreedy(epsilon=0.1, num_actions=2)
    parametrised_qs = BasicQs(initial_value=0.0, num_states=3, num_actions=2,
                              learning_rate=0.8, discount_rate=0.9)
    episode = SarsaEpisode(im, parametrised_qs, epsilon_greedy_policy, 1000)
    run = SarsaRun(1000, episode)
    return run, parametrised_qs

def homing_robot():
    hr = HomingRobot(10, 10, (5, 5))
    epsilon_greedy_policy = EpsilonGreedy(epsilon=0.1, num_actions=hr.num_actions)
    softmax_decay_policy = SoftMaxDecay(num_actions=hr.num_actions, tau=20)
    parametrised_qs = BasicQs(initial_value=0.0, num_states=hr.num_states,
                              num_actions=hr.num_actions, learning_rate=0.8,
                              discount_rate=0.9)
    episode = SarsaEpisode(hr, parametrised_qs, softmax_decay_policy, 20)
    run = SarsaRun(1000, episode)
    return run, parametrised_qs

# if __name__ == '__main__':
#     run, parametrised_qs = homing_robot()
#     run.run()
#     print(parametrised_qs.parameters)

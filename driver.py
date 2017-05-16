import numpy as np
import coloredlogs
import cProfile as profile
import pstats

from assignment.environment import *
from assignment.sarsa.core import *
from assignment.sarsa.policy import *
from assignment.sarsa.qmodel import *

coloredlogs.install(level='INFO')

hr = HomingRobot(10, 10, (5, 5))
im = ImageMonkey()

parametrised_qs = BasicQs(initial_value=0.0, num_states=3, num_actions=2,
                          learning_rate=0.8, discount_rate=0.9)
epsilon_greedy_policy = EpsilonGreedy(epsilon=0.1, num_actions=2)
episode_factory = lambda: SarsaEpisode(ImageMonkey(), parametrised_qs,
                                       epsilon_greedy_policy, 1000)
run = SarsaRun(1000, episode_factory)

if __name__ == '__main__':
    run.run()
    print(parametrised_qs.parameters)

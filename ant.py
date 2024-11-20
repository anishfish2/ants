import functools

import gymnasium
from gymnasium.spaces import Discrete
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
 
import numpy


class ant():
    def __init__(self):
        self.location = {'x' : 0, 'y' : 0}
        self.current_angle = 0

    def update_angle(self, angle_diff):
        self.current_angle += angle_diff

    def update_location(self, distance):
        self.location['x'] += distance * numpy.cos(self.current_angle)
        self.location['y'] += distance * numpy.sin(self.current_angle)
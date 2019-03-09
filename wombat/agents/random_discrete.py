import numpy as np


class RandomDiscrete:
    def __init__(self, num_possible_actions):
        self.num_possible_actions = num_possible_actions


    def act(self, **kwargs):
        return np.random.randint(self.num_possible_actions)


    def train_step(self, **kwargs):
        pass
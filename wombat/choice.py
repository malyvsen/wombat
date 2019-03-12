import numpy as np


def best(expected_rewards):
    '''Find the best action given the expected rewards of taking each action'''
    return np.argmax(expected_rewards)


def epsilon_greedy(epsilon):
    '''Return a function which picks a random action with epsilon probability, and the best action otherwise'''
    def policy(expected_rewards):
        return np.random.randint(len(expected_rewards)) if np.random.random() < epsilon else best(expected_rewards)
    return policy

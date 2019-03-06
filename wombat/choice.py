import numpy as np


def expected_rewards(model, tf_session, observation, num_possible_actions):
    '''Run model to calculate the expected discounted rewards for each possible action'''
    feed_dict = {
        model.observations: [observation] * num_possible_actions,
        model.actions: list(range(num_possible_actions))}
    return tf_session.run(model.expected_rewards, feed_dict=feed_dict)


def best(expected_rewards, **kwargs):
    '''Find the best action given the expected rewards of taking each action'''
    return np.argmax(expected_rewards)


def epsilon_greedy(epsilon):
    def policy(expected_rewards, **kwargs):
        return np.random.randint(len(expected_rewards)) if np.random.random() < epsilon else best(expected_rewards)
    return policy

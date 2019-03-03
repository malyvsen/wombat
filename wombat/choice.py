import numpy as np


def best_action(expected_rewards):
    '''Find the best action given the expected rewards of taking each action'''
    return np.argmax(expected_rewards)


def expected_rewards(model, tf_session, observation, num_possible_actions):
    '''Run model to calculate the expected discounted rewards for each possible action'''
    feed_dict = {
        model.observations: [observation] * num_possible_actions,
        model.actions: list(range(num_possible_actions))}
    return tf_session.run(model.expected_rewards, feed_dict=feed_dict)

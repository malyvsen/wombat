import numpy as np
import tensorflow as tf
from wombat.choice import epsilon_greedy


class DQN:
    '''A DQN-like agent sample implementation'''
    def __init__(
        self,
        num_possible_actions,
        observations,
        actions,
        expected_rewards,
        true_rewards,
        optimize,
        discount=0.99,
        action_chooser=epsilon_greedy(0.1)):
        '''Create a DQN agent given model as tensors'''
        self.num_possible_actions = num_possible_actions
        self.observations = observations
        self.actions = actions
        self.expected_rewards = expected_rewards
        self.true_rewards = true_rewards
        self.optimize = optimize
        self.discount = discount
        self.action_chooser = action_chooser


    def act(self, session, episode):
        return self.action_chooser(expected_rewards=self.eval_expected_rewards(session, episode.observations[-1]))


    def train_step(self, session, episode, step_id):
        expected_future_rewards = self.eval_expected_rewards(session=session, observation=episode.observations[step_id + 1])
        done = (step_id + 1 == len(episode)) if episode.finished else False
        discounted_reward = episode.rewards[step_id] + (0 if done else self.discount * np.max(expected_future_rewards))
        feed_dict = {
            self.observations: [episode.observations[step_id]],
            self.actions: [episode.actions[step_id]],
            self.true_rewards: [discounted_reward]}
        session.run(self.optimize, feed_dict=feed_dict)


    def eval_expected_rewards(self, session, observation):
        '''Run model to calculate the expected discounted rewards for each possible action'''
        feed_dict = {
            self.observations: [observation] * self.num_possible_actions,
            self.actions: list(range(self.num_possible_actions))}
        return session.run(self.expected_rewards, feed_dict=feed_dict)

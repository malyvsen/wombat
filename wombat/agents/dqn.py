import numpy as np
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


    def act(self, episode, session):
        return self.action_chooser(expected_rewards=self.eval_expected_rewards(observation=episode.steps[-1].observation, session=session))


    def train(self, steps, session):
        '''Train model on given steps'''
        for step in reversed(steps): # reverse to avoid fitting to things that will soon change
            if step.action is None:
                continue
            expected_future_rewards = self.eval_expected_rewards(observation=step.observation, session=session)
            discounted_reward = step.reward + (0 if step.done else self.discount * np.max(expected_future_rewards))
            feed_dict = {
                self.observations: [step.context[-1].observation],
                self.actions: [step.action],
                self.true_rewards: [discounted_reward]}
            session.run(self.optimize, feed_dict=feed_dict)


    def eval_expected_rewards(self, observation, session):
        '''Run model to calculate the expected discounted rewards for each possible action'''
        feed_dict = {
            self.observations: [observation] * self.num_possible_actions,
            self.actions: list(range(self.num_possible_actions))}
        return session.run(self.expected_rewards, feed_dict=feed_dict)

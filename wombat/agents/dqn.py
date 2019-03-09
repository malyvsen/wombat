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


    def train(self, episode, session, start_step=1, end_step=None):
        '''Train model on steps from given episode'''
        end_step = end_step if end_step is not None else len(episode)
        for step_id in reversed(range(start_step, end_step)): # reverse to avoid fitting to things that will soon change
            if episode.steps[step_id].action is None:
                continue
            expected_future_rewards = self.eval_expected_rewards(observation=episode.steps[step_id].observation, session=session)
            discounted_reward = episode.steps[step_id].reward + (0 if episode.steps[step_id].done else self.discount * np.max(expected_future_rewards))
            feed_dict = {
                self.observations: [episode.steps[step_id - 1].observation],
                self.actions: [episode.steps[step_id].action],
                self.true_rewards: [discounted_reward]}
            session.run(self.optimize, feed_dict=feed_dict)


    def eval_expected_rewards(self, observation, session):
        '''Run model to calculate the expected discounted rewards for each possible action'''
        feed_dict = {
            self.observations: [observation] * self.num_possible_actions,
            self.actions: list(range(self.num_possible_actions))}
        return session.run(self.expected_rewards, feed_dict=feed_dict)

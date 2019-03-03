import numpy as np
import wombat.choice as choice


class EpisodeReplay:
    def __init__(self, initial_observation, num_possible_actions):
        self.observations = [initial_observation]
        self.actions = []
        self.rewards = []
        self.finished = False
        self.num_possible_actions = num_possible_actions


    def register_step(self, observation, action, reward, done):
        '''Register step in which action was taken to yield observation and reward'''
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        if done:
            self.finished = True


    def train(self, model, tf_session, discount, learning_rate, start_step=0, end_step=None):
        '''Train model on the steps from this episode'''
        if end_step is None:
            end_step = len(self)
        for step in reversed(range(start_step, end_step)): # reverse so that we don't fit to things that will soon be modified
            expected_rewards = choice.expected_rewards(
                model=model,
                tf_session=tf_session,
                observation=self.observations[step + 1],
                num_possible_actions=self.num_possible_actions)
            done = (step + 1 == len(self.actions)) if self.finished else False
            discounted_reward = self.rewards[step] + (0 if done else discount * np.max(expected_rewards))
            feed_dict = {
                model.observations: [self.observations[step]],
                model.actions: [self.actions[step]],
                model.target_expected_rewards: [discounted_reward],
                model.learning_rate: learning_rate}
            tf_session.run(model.optimize, feed_dict=feed_dict)


    def total_reward(self):
        return np.sum(self.rewards)


    def __len__(self):
        '''
        The number of registered steps
        Note that the number of registered observations is one greater than this, due to environments producing initial observations
        '''
        return len(self.rewards)

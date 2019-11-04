import numpy as np
from wombat.choice import epsilon_greedy


class DQN:
    '''A sample DQN agent'''
    def __init__(
        self,
        get_expected_rewards,
        optimize,
        discount=.99,
        choose_action=epsilon_greedy(0.1)):
        '''Create a DQN agent given an interface to a model.

        Parameters
        ----------
        get_expected_rewards : observation => [float]
            Function which calculates the expected discounted rewards for each possible action.
        optimize : (observation, action : int, target_discounted_reward : float) => loss : float
            Function used to train the model on a step-by-step basis.
        discount : float, optional
        choose_action : [float] => int, optional
            Function used to select action given list of expected rewards per action, should return index into list.'''
        self.get_expected_rewards = get_expected_rewards
        self.optimize = optimize
        self.discount = discount
        self.choose_action = choose_action


    def act(self, steps):
        return self.choose_action(self.get_expected_rewards(steps[-1].observation))


    def train(self, steps):
        losses = []
        for step in reversed(steps): # reverse to avoid fitting to things that will soon change
            if step.action is None:
                continue
            expected_future_rewards = self.get_expected_rewards(step.observation)
            discounted_reward = step.reward + (0 if step.done else self.discount * np.max(expected_future_rewards))
            loss = self.optimize(step.context[-1].observation, step.action, discounted_reward)
            losses.append(loss)
        return np.mean(losses)

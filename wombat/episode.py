import numpy as np
import wombat.choice as choice


class Episode:
    def __init__(self, initial_observation=None):
        self.observations = [initial_observation] if initial_observation is not None else []
        self.actions = []
        self.rewards = []
        self.finished = False


    def run(self, agent, session, environment):
        '''
        Generator that runs model in OpenAI-gym-like environment until done
        Yield tuples of (observation, reward, done, info, action)
        '''
        if len(self.observations) == 0:
            self.observations.append(environment.reset())
        while True:
            action = agent.act(session=session, episode=self)
            observation, reward, done, info = environment.step(action)
            self.register_step(observation, reward, done, action)
            yield observation, reward, done, info, action
            if done:
                break


    def register_step(self, observation, reward, done, action):
        '''
        Register step in which action was taken to yield observation and reward
        You usually don't need to do this - Episode.run handles this for you
        '''
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        if done:
            self.finished = True


    def train(self, agent, session, start_step=0, end_step=None):
        '''Train agent on the steps from this episode'''
        if end_step is None:
            end_step = len(self)
        for step_id in reversed(range(start_step, end_step)): # reverse so that we don't fit to things that will soon be modified
            agent.train_step(session=session, episode=self, step_id=step_id)


    def total_reward(self):
        return np.sum(self.rewards)


    def __len__(self):
        '''
        The number of registered steps
        Note that the number of registered observations is one greater than this, due to environments producing initial observations
        '''
        return len(self.rewards)

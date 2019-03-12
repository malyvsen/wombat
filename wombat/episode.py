from wombat.context import Context
from wombat.step import Step


class Episode:
    def __init__(self, steps=None):
        self.steps = steps if steps is not None else []


    def run(self, agent, environment):
        '''Step generator that runs model in OpenAI-gym-like environment until done'''
        if len(self) == 0:
            self.steps.append(Step(observation=environment.reset(), episode=self))
        while True:
            action = agent.act(episode=self)
            step = Step.run(environment=environment, action=action, episode=self)
            self.steps.append(step)
            yield step
            if step.done:
                break


    def total_reward(self):
        return sum(step.reward for step in self.steps)


    def __len__(self):
        '''
        The number of registered steps
        The first step corresponds to the initial observation of the environment, so the number of actions taken is one less than this
        '''
        return len(self.steps)

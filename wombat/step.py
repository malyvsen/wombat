from wombat.context import Context


class Step:
    '''
    Stores information about a single step
    step.action was taken to yield step.observation, step.reward, step.done, step.info
    '''
    def __init__(self, observation, reward=0, done=False, info={}, action=None, episode=None):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info
        self.action = action
        self.episode = episode
        self.context = None if episode is None else Context(steps=episode.steps, offset=len(episode.steps))


    @classmethod
    def run(cls, environment, action, episode=None):
        '''Perform action in environment and return result'''
        observation, reward, done, info = environment.step(action)
        return cls(observation=observation, reward=reward, done=done, info=info, action=action, episode=episode)

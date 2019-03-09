class Step:
    '''
    Stores information about a single step
    step.action was taken to yield step.observation, step.reward, step.done, step.info
    '''
    def __init__(self, observation, reward=0, done=False, info={}, action=None):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info
        self.action = action


    @classmethod
    def run(cls, environment, action):
        '''Perform action in environment and return result'''
        observation, reward, done, info = environment.step(action)
        return cls(observation=observation, reward=reward, done=done, info=info, action=action)

class Context:
    '''
    The entire context of a step
    A lightweight interface to the step's surrounding steps
    context[0] is the given step, context[-1] is the previous one, context[+1] is the next one etc
    '''
    def __init__(self, steps, offset=0):
        self.steps = steps
        self.offset = offset # no assertion - sometimes we add a step to a list after initializing its context


    def __getitem__(self, key):
        index = key + self.offset
        assert 0 <= index, 'cannot get step before start of context'
        assert index < len(self.steps), 'cannot get step after end of context'
        return self.steps[index]

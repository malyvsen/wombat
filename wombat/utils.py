import inspect


def careful_call(function, **kwargs):
    '''
    Call function providing only those keyword args that it explicitly takes, and return what it returned
    If function proves not to be a function, just return what was passed
    '''
    if not callable(function):
        return function
    argspec = inspect.getfullargspec(function)
    accepted_args = argspec.args + argspec.kwonlyargs
    args_to_pass = {arg: kwargs[arg] for arg in kwargs if arg in accepted_args}
    return function(**args_to_pass)

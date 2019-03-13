import inspect


def smart_call(obj, **kwargs):
    '''
    If obj is callable, call it providing only those keyword args that it explicitly takes, and return obj it returned
    Otherwise, if it is an iterable, recursively iterate over it
    Otherwise, just return what was passed
    '''
    if not callable(obj):
        try:
            return type(obj)([smart_call(sub_obj, **kwargs) for sub_obj in obj]) # attempt to re-create original type
        except TypeError:
            return obj
    argspec = inspect.getfullargspec(obj)
    accepted_args = argspec.args + argspec.kwonlyargs
    args_to_pass = {arg: kwargs[arg] for arg in kwargs if arg in accepted_args}
    return obj(**args_to_pass)

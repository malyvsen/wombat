import random
from wombat.utils import smart_call
from wombat.train.weighters import uniform


def online(max_last_steps=1):
    '''Train agent on the last few steps of the last episode'''
    def result(agent, episodes):
        smart_call(agent.train, steps=episodes[-1].steps[-max_last_steps : ])
    return result


def offline(num_replays=4, max_steps_per_replay=None, weighter=uniform()):
    '''Randomly (with replacement) pick episode replays and train agent on them'''
    def result(agent, episodes):
        chosen_replays = random.choices(episodes, k=num_replays, weights=smart_call(weighter, episodes=episodes))
        for replay in chosen_replays:
            if max_steps_per_replay is not None:
                num_steps_to_train_on = min(len(replay), max_steps_per_replay)
            else:
                num_steps_to_train_on = len(replay)
            start_step = random.randrange(len(replay) - num_steps_to_train_on + 1)
            end_step = start_step + num_steps_to_train_on
            replay.loss = smart_call(agent.train, steps=replay.steps[start_step : end_step])
    return result

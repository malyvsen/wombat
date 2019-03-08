import numpy as np
import random
from tqdm import trange
from wombat.episode import Episode
from wombat.choice import epsilon_greedy


def train(
    agent,
    session,
    environment,
    num_episodes=256,
    online_steps_per_replay=16,
    max_replay_steps=None,
    num_replays_after_completed_episode=4):
    '''
    Train agent on given OpenAI-gym-like environment
    Return a list of replays for all episodes
    '''

    episodes = []
    total_passed_steps = 0

    for episode_id in trange(num_episodes):
        episode = Episode()
        episodes.append(episode)

        for step in episode.run(agent=agent, session=session, environment=environment):
            episode.train(agent=agent, session=session, start_step=len(episode) - 1) # online training - on last step only
            if total_passed_steps % online_steps_per_replay == 0 and len(episodes) > 0:
                train_on_replays(agent=agent, session=session, episode_replays=episodes, num_replays=1, max_replay_steps=max_replay_steps)
            total_passed_steps += 1

        train_on_replays(
            agent=agent,
            session=session,
            episode_replays=episodes,
            num_replays=num_replays_after_completed_episode,
            max_replay_steps=max_replay_steps)

    return episodes


def test(agent, session, environment, num_episodes=4):
    '''
    Test agent on given OpenAI-gym-like environment
    Return a list of replays for all episodes
    '''
    episodes = []

    for episode_id in trange(num_episodes):
        episode = Episode(initial_observation=environment.reset())
        episodes.append(episode)
        environment.render()
        for step in episode.run(agent=agent, session=session, environment=environment):
            environment.render()

    return episodes


def train_on_replays(agent, session, episode_replays, num_replays, max_replay_steps=None):
    '''Randomly (with replacement) pick episode replays and train on them'''
    replays_to_train_on = random.choices(episode_replays, k=num_replays)
    for episode_replay in replays_to_train_on:
        if max_replay_steps is not None:
            num_steps_to_train_on = min(len(episode_replay), max_replay_steps)
        else:
            num_steps_to_train_on = len(episode_replay)
        start_step = np.random.randint(0, len(episode_replay) - num_steps_to_train_on + 1)
        end_step = start_step + num_steps_to_train_on
        episode_replay.train(
            agent=agent,
            session=session,
            start_step=start_step,
            end_step=end_step)

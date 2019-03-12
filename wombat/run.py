import numpy as np
import random
from tqdm import trange
from wombat.episode import Episode
from wombat.choice import epsilon_greedy


def train(
    agent,
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

        for step in episode.run(agent=agent, environment=environment):
            agent.train(steps=episode.steps[-1:]) # online training - on last step only
            if total_passed_steps % online_steps_per_replay == 0 and len(episodes) > 0:
                train_on_replays(agent=agent, episode_replays=episodes, num_replays=1, max_replay_steps=max_replay_steps)
            total_passed_steps += 1

        train_on_replays(
            agent=agent,
            episode_replays=episodes,
            num_replays=num_replays_after_completed_episode,
            max_replay_steps=max_replay_steps)

    return episodes


def test(agent, environment, num_episodes=4):
    '''
    Test agent on given OpenAI-gym-like environment
    Return a list of replays for all episodes
    '''
    episodes = []

    for episode_id in trange(num_episodes):
        episode = Episode()
        episodes.append(episode)
        for step in episode.run(agent=agent, environment=environment):
            environment.render()

    return episodes


def train_on_replays(agent, episode_replays, num_replays, max_replay_steps=None):
    '''Randomly (with replacement) pick episode replays and train on them'''
    chosen_replays = random.choices(episode_replays, k=num_replays)
    for episode_replay in chosen_replays:
        if max_replay_steps is not None:
            num_steps_to_train_on = min(len(episode_replay), max_replay_steps)
        else:
            num_steps_to_train_on = len(episode_replay)
        start_step = np.random.randint(0, len(episode_replay) - num_steps_to_train_on + 1)
        end_step = start_step + num_steps_to_train_on
        agent.train(steps=episode_replay.steps[start_step : end_step])

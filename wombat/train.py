import numpy as np
import random
from tqdm import trange
from wombat.episode_replay import EpisodeReplay
import wombat.choice as choice


def run(model, tf_session, environment, test=False, num_episodes=None, epsilon=None, discount=.99, online_learning_rate=2e-3, offline_learning_rate=2e-3, learning_decay=.97, offline_per_online=16):
    '''
    Run model in testing or training mode on given OpenAI-gym-like environment
    Return a list of replays for all episodes
    '''
    if num_episodes is None:
        num_episodes = 4 if test else 256
    if epsilon is None:
        epsilon = 0 if test else .1

    episode_replays = []
    rolling_average_reward = 0

    for episode in trange(num_episodes):
        observation = environment.reset()
        replay = EpisodeReplay(initial_observation=observation, num_possible_actions=environment.action_space.n)
        done = False

        while True:
            best_action = choice.best_action(choice.expected_rewards(model, tf_session, observation, num_possible_actions=environment.action_space.n))
            chosen_action = environment.action_space.sample() if np.random.random() < epsilon else best_action

            if test:
                environment.render()
            elif len(replay.rewards) > 0:
                learning_rate = online_learning_rate * learning_decay ** rolling_average_reward
                replay.train(
                    model=model,
                    tf_session=tf_session,
                    discount=discount,
                    learning_rate=learning_rate,
                    start_step=len(replay.rewards) - 1) # train on last step only

            if done:
                episode_replays.append(replay)
                rolling_average_reward = rolling_average_reward * learning_decay + replay.total_reward() * (1 - learning_decay)
                break

            observation, reward, done, info = environment.step(chosen_action)
            replay.register_step(observation=observation, action=chosen_action, reward=reward, done=done)

        if not test:
            # offline training
            learning_rate = offline_learning_rate * learning_decay ** rolling_average_reward
            train_offline(
                model=model,
                tf_session=tf_session,
                episode_replays=episode_replays,
                num_episodes=offline_per_online,
                discount=discount,
                learning_rate=learning_rate)

    environment.close()
    return episode_replays


def train_offline(model, tf_session, episode_replays, num_episodes, discount, learning_rate):
    '''Randomly (with replacement) pick episode replays and train on them'''
    replays_to_train_on = random.choices(episode_replays, k=num_episodes)
    for episode_replay in replays_to_train_on:
        episode_replay.train(model=model, tf_session=tf_session, discount=discount, learning_rate=learning_rate)

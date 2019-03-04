import numpy as np
import random
from tqdm import trange
from wombat.episode_replay import EpisodeReplay
import wombat.choice as choice


def train(
    model,
    tf_session,
    environment,
    num_episodes=256,
    epsilon=.1,
    discount=.99,
    online_learning_rate=2e-3,
    replay_learning_rate=2e-3,
    learning_decay=.97,
    online_steps_per_replay=16,
    max_replay_steps=None,
    num_replays_after_completed_episode=4):
    '''
    Train model on given OpenAI-gym-like environment
    Return a list of replays for all episodes
    '''

    episode_replays = []
    total_passed_steps = 0
    rolling_average_reward = 0

    for episode in trange(num_episodes):
        observation = environment.reset()
        current_replay = EpisodeReplay(initial_observation=observation, num_possible_actions=environment.action_space.n)
        done = False

        while True:
            best_action = choice.best_action(choice.expected_rewards(model, tf_session, observation, num_possible_actions=environment.action_space.n))
            chosen_action = environment.action_space.sample() if np.random.random() < epsilon else best_action

            if total_passed_steps % online_steps_per_replay == 0 and len(episode_replays) > 0:
                train_on_replays(
                    model=model,
                    tf_session=tf_session,
                    episode_replays=episode_replays,
                    num_replays=1,
                    discount=discount,
                    learning_rate=replay_learning_rate * learning_decay ** rolling_average_reward,
                    max_replay_steps=max_replay_steps)
            if len(current_replay) > 0:
                current_replay.train(
                    model=model,
                    tf_session=tf_session,
                    discount=discount,
                    learning_rate=online_learning_rate * learning_decay ** rolling_average_reward,
                    start_step=len(current_replay) - 1) # train on last step only

            if done:
                episode_replays.append(current_replay)
                rolling_average_reward = rolling_average_reward * learning_decay + current_replay.total_reward() * (1 - learning_decay)
                break

            observation, reward, done, info = environment.step(chosen_action)
            current_replay.register_step(observation=observation, action=chosen_action, reward=reward, done=done)
            total_passed_steps += 1

        train_on_replays(
            model=model,
            tf_session=tf_session,
            episode_replays=episode_replays,
            num_replays=num_replays_after_completed_episode,
            discount=discount,
            learning_rate=replay_learning_rate * learning_decay ** rolling_average_reward,
            max_replay_steps=max_replay_steps)

    environment.close()
    return episode_replays


def test(
    model,
    tf_session,
    environment,
    num_episodes=4,
    epsilon=0.05):
    '''
    Test model on given OpenAI-gym-like environment
    Return a list of replays for all episodes
    '''

    episode_replays = []

    for episode in trange(num_episodes):
        observation = environment.reset()
        current_replay = EpisodeReplay(initial_observation=observation, num_possible_actions=environment.action_space.n)
        done = False

        while not done:
            environment.render()

            best_action = choice.best_action(choice.expected_rewards(model, tf_session, observation, num_possible_actions=environment.action_space.n))
            chosen_action = environment.action_space.sample() if np.random.random() < epsilon else best_action
            observation, reward, done, info = environment.step(chosen_action)
            current_replay.register_step(observation=observation, action=chosen_action, reward=reward, done=done)

        episode_replays.append(current_replay)

    environment.close()
    return episode_replays


def train_on_replays(model, tf_session, episode_replays, num_replays, discount, learning_rate, max_replay_steps=None):
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
            model=model,
            tf_session=tf_session,
            discount=discount,
            learning_rate=learning_rate,
            start_step=start_step,
            end_step=end_step)

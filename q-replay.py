#%% imports, setup
import gym
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt


env_name = 'CartPole-v0'
env = gym.make(env_name)


#%% model
class Model:
    observations = tf.placeholder(tf.float32, shape=(None, env.observation_space.shape[0]))
    actions = tf.placeholder(tf.int32, shape=(None,))
    actions_one_hot = tf.one_hot(actions, env.action_space.n)
    inputs = tf.concat((observations, actions_one_hot), -1)
    layers = [inputs]

    layers.append(tf.layers.dense(layers[-1], 32, activation=tf.nn.relu))

    expected_rewards = tf.layers.dense(layers[-1], 1)
    layers.append(expected_rewards)

    target_expected_rewards = tf.placeholder(tf.float32, shape=(None,))
    loss = tf.losses.mean_squared_error(tf.reshape(target_expected_rewards, (-1, 1)), expected_rewards)

    learning_rate = tf.placeholder(tf.float32, shape=())
    optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss)


#%% tensorflow init
session = tf.Session()
session.run(tf.global_variables_initializer())


#%% utils
def get_expected_rewards(observation):
    '''Run model to calculate the expected discounted rewards for each possible action'''
    feed_dict = {
        Model.observations: [observation] * env.action_space.n,
        Model.actions: list(range(env.action_space.n))}
    return session.run(Model.expected_rewards, feed_dict=feed_dict)


def get_best_action(expected_rewards):
    return np.argmax(expected_rewards)


def train_step(observation, action, best_expected_rewards, reward, done, learning_rate, discount):
    discounted_rewards = reward + (0 if done else discount * best_expected_rewards)
    feed_dict = {
        Model.observations: [observation],
        Model.actions: [action],
        Model.target_expected_rewards: [discounted_rewards],
        Model.learning_rate: learning_rate}
    session.run(Model.optimize, feed_dict=feed_dict)


#%% replay buffer-related
class EpisodeReplay:
    def __init__(self, original_observation):
        self.observations = [original_observation]
        self.actions = []
        self.rewards = []
        self.is_finished = False


    def register_step(self, observation, action, reward, done):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        if done:
            self.is_finished = True


    def train(self, learning_rate, discount, start_step=0):
        for step in reversed(range(start_step, len(self.actions))): # reverse so that we don't fit to things that will soon be modified
            expected_rewards = get_expected_rewards(observation=self.observations[step + 1])
            done = (step + 1 == len(self.actions)) if self.is_finished else False
            train_step(
                observation=self.observations[step],
                action=self.actions[step],
                best_expected_rewards=np.max(expected_rewards),
                reward=self.rewards[step],
                done=done,
                learning_rate=learning_rate,
                discount=discount)


    def get_total_reward(self):
        return np.sum(self.rewards)


#%% training/testing
def run(test=False, num_episodes=None, discount=.99, epsilon=None, online_learning_rate=2e-3, offline_learning_rate=2e-3, learning_rate_decay=.97, offline_episodes=16):
    if num_episodes is None:
        num_episodes = 4 if test else 256
    if epsilon is None:
        epsilon = 0 if test else .1

    env = gym.make(env_name)
    episode_replays = []
    rolling_average_reward = 0

    for episode in range(num_episodes):
        observation = env.reset()
        replay = EpisodeReplay(observation)
        done = False

        while True:
            best_action = get_best_action(get_expected_rewards(observation))
            chosen_action = env.action_space.sample() if np.random.random() < epsilon else best_action

            if test:
                env.render()
            elif len(replay.rewards) > 0:
                learning_rate = online_learning_rate * learning_rate_decay ** rolling_average_reward
                replay.train(learning_rate=learning_rate, discount=discount, start_step=len(replay.rewards) - 1) # train on last step only

            if done:
                episode_replays.append(replay)
                print(f'Episode {episode} finished, total reward: {replay.get_total_reward()}')
                rolling_average_reward = rolling_average_reward * learning_rate_decay + replay.get_total_reward() * (1 - learning_rate_decay)
                break

            observation, reward, done, info = env.step(chosen_action)
            replay.register_step(observation=observation, action=chosen_action, reward=reward, done=done)

        if not test:
            replays_to_train_on = random.choices(episode_replays, k=offline_episodes)
            for episode_replay in replays_to_train_on:
                learning_rate = offline_learning_rate * learning_rate_decay ** rolling_average_reward
                episode_replay.train(learning_rate=learning_rate, discount=discount)

    env.close()
    return episode_replays


#%% demo
if __name__ == '__main__':
    episode_replays = run(test=False)
    plt.plot([episode_replay.get_total_reward() for episode_replay in episode_replays])
    plt.xlabel('episode')
    plt.ylabel('epsiode reward')
    plt.show()
    run(test=True)

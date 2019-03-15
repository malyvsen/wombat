import gym
import tensorflow as tf
import numpy as np
import wombat


env = gym.make('CartPole-v0')


def test_dqn():
    observations = tf.placeholder(tf.float32, shape=(None, env.observation_space.shape[0]))
    actions = tf.placeholder(tf.int32, shape=(None,))
    actions_one_hot = tf.one_hot(actions, env.action_space.n)

    inputs = tf.concat((observations, actions_one_hot), -1)
    intermediate = tf.layers.Dense(32, activation=tf.nn.relu)(inputs)
    expected_rewards = tf.layers.Dense(1)(intermediate)

    true_rewards = tf.placeholder(tf.float32, shape=(None,))
    loss = tf.losses.mean_squared_error(tf.reshape(true_rewards, (-1, 1)), expected_rewards)
    optimize = tf.train.AdamOptimizer(2e-3).minimize(loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        agent = wombat.agents.DQN(env.action_space.n, observations, actions, expected_rewards, true_rewards, loss, optimize, session)
        training_episodes = wombat.run(agent=agent, environment=env, num_episodes=192, per_step=wombat.train.online(), per_episode=wombat.train.offline())
        assert np.mean(list(episode.total_reward() for episode in training_episodes[-64:])) > 100


def test_random_discrete():
    agent = wombat.agents.RandomDiscrete(env.action_space.n)
    episodes = wombat.run(agent=agent, environment=env, num_episodes=256)
    assert np.mean(list(episode.total_reward() for episode in episodes)) < 25

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
    intermediate = tf.layers.dense(inputs, 32, activation=tf.nn.relu)
    expected_rewards = tf.layers.dense(intermediate, 1)

    true_rewards = tf.placeholder(tf.float32, shape=(None,))
    loss = tf.losses.mean_squared_error(tf.reshape(true_rewards, (-1, 1)), expected_rewards)
    optimize = tf.train.AdamOptimizer(2e-3).minimize(loss)

    agent = wombat.agents.DQN(env.action_space.n, observations, actions, expected_rewards, true_rewards, optimize)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        training_episodes = wombat.train(agent=agent, session=session, environment=env, num_episodes=128, online_steps_per_replay=16, max_replay_steps=None, num_replays_after_completed_episode=4)
        assert np.mean(list(episode.total_reward() for episode in training_episodes[-32:])) > 100


def test_random_discrete():
    agent = wombat.agents.RandomDiscrete(env.action_space.n)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        training_episodes = wombat.train(agent=agent, session=session, environment=env)
        assert np.mean(list(episode.total_reward() for episode in training_episodes)) < 25

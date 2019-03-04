import gym
import tensorflow as tf
from wombat.train import run
import matplotlib.pyplot as plt


env_name = 'CartPole-v0'
env = gym.make(env_name)


class Model:
    observations = tf.placeholder(tf.float32, shape=(None, env.observation_space.shape[0])) # obligatory for wombat to work
    actions = tf.placeholder(tf.int32, shape=(None,)) # obligatory for wombat to work
    actions_one_hot = tf.one_hot(actions, env.action_space.n)
    inputs = tf.concat((observations, actions_one_hot), -1)
    intermediate = tf.layers.dense(inputs, 32, activation=tf.nn.relu)
    expected_rewards = tf.layers.dense(intermediate, 1)
    target_expected_rewards = tf.placeholder(tf.float32, shape=(None,)) # obligatory for wombat to work
    loss = tf.losses.mean_squared_error(tf.reshape(target_expected_rewards, (-1, 1)), expected_rewards)

    learning_rate = tf.placeholder(tf.float32, shape=()) # obligatory for wombat to work
    optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss) # obligatory for wombat to work


with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    episode_replays = run(model=Model, tf_session=session, environment=env, test=False)
    plt.plot([episode_replay.total_reward() for episode_replay in episode_replays])
    plt.xlabel('episode')
    plt.ylabel('episode reward')
    plt.show()
    run(model=Model, tf_session=session, environment=env, test=True)

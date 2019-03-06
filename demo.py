import gym
import tensorflow as tf
from wombat.run import train, test
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
    optimize = tf.train.AdamOptimizer(2e-3).minimize(loss) # obligatory for wombat to work


with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    training_episodes = train(model=Model, tf_session=session, environment=env)
    plt.plot([episode.total_reward() for episode in training_episodes])
    plt.xlabel('episode')
    plt.ylabel('episode reward')
    plt.show()
    test(model=Model, tf_session=session, environment=env)

import gym
import tensorflow as tf
import wombat
import matplotlib.pyplot as plt


env_name = 'CartPole-v0'
env = gym.make(env_name)


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
    training_episodes = wombat.train(agent=agent, session=session, environment=env)
    plt.plot([episode.total_reward() for episode in training_episodes])
    plt.xlabel('episode')
    plt.ylabel('episode reward')
    plt.show()
    wombat.test(agent=agent, session=session, environment=env)

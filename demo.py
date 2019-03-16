import gym
import tensorflow as tf
import wombat
import matplotlib.pyplot as plt


# wombat supports OpenAI-gym-like environments out of the box
# custom environments must have step(), reset() and close() functioning like in gym
env = gym.make('CartPole-v0')


# tensorflow is used here for demonstration purposes, but wombat doesn't require it
# the sample implementation of DQN does, however, assume that tensorflow is being used
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
    # a wombat agent is where the decision-making and training take place
    # wombat makes creating custom agents easy, just define act() and train() methods - see wombat.agents.DQN for an example
    agent = wombat.agents.DQN(env.action_space.n, observations, actions, expected_rewards, true_rewards, loss, optimize, session)
    training_episodes = wombat.run(agent, env, 256, per_step=wombat.train.online(), per_episode=wombat.train.offline(weighter=wombat.train.weighters.prioritized(temperature=8)))
    plt.plot([episode.total_reward() for episode in training_episodes])
    plt.xlabel('episode')
    plt.ylabel('episode reward')
    plt.show()
    wombat.run(agent=agent, environment=env, num_episodes=4, per_step=env.render)

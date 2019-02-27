#%% imports, setup
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


env_name = 'CartPole-v0'
env = gym.make(env_name)


#%% model
class Model:
    inputs = tf.placeholder(tf.float32, shape=(None, env.observation_space.shape[0]))
    layers = [inputs]

    layers.append(tf.layers.dense(layers[-1], 32, activation=tf.nn.relu))

    expected_rewards = tf.layers.dense(layers[-1], env.action_space.n)
    layers.append(expected_rewards)

    target_expected_rewards = tf.placeholder(tf.float32, shape=(None, env.action_space.n))
    loss_weights = tf.placeholder(tf.float32, shape=(None, env.action_space.n))
    loss = tf.losses.mean_squared_error(target_expected_rewards, expected_rewards, weights=loss_weights)

    learning_rate = tf.placeholder(tf.float32, shape=())
    optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss)


#%% utils
def sample_action(distribution):
    return np.sample(np.arange(env.action_space.n), p=model_outputs)


def one_hot(val, max_val):
    return np.array([i == val for i in range(max_val)], dtype=np.int32)


#%% training/testing procedure
session = tf.Session()
session.run(tf.global_variables_initializer())


def run(test=False, num_episodes=None, discount=.99, epsilon=None, base_learning_rate=2e-3, learning_rate_decay=.97):
    if num_episodes is None:
        num_episodes = 4 if test else 4096
    if epsilon is None:
        epsilon = 0 if test else .2

    env = gym.make(env_name)
    rolling_average_reward = 0
    reward_history = []

    for episode in range(num_episodes):
        state = env.reset()
        prev_state = None
        prev_action = None
        reward = None
        done = False
        episode_reward = 0
        while True:
            expected_rewards = session.run(Model.expected_rewards, feed_dict={Model.inputs: [state]})[0]
            best_action = np.argmax(expected_rewards)
            chosen_action = np.random.randint(env.action_space.n) if np.random.random() < epsilon else best_action

            if test:
                env.render()
            elif prev_state is not None:
                discounted_rewards = reward + (0 if done else discount * expected_rewards[chosen_action])
                prev_action_one_hot = one_hot(prev_action, env.action_space.n)
                target_expected_rewards = prev_action_one_hot * discounted_rewards
                learning_rate = base_learning_rate * learning_rate_decay ** max(0, rolling_average_reward)
                feed_dict = {
                    Model.inputs: [prev_state],
                    Model.target_expected_rewards: [target_expected_rewards],
                    Model.loss_weights: [prev_action_one_hot],
                    Model.learning_rate: learning_rate}
                session.run(Model.optimize, feed_dict=feed_dict)
            prev_state = state
            prev_action = chosen_action

            if done:
                rolling_average_reward = rolling_average_reward * learning_rate_decay + episode_reward * (1 - learning_rate_decay)
                reward_history.append(episode_reward)
                print(f'Episode {episode} finished, total reward: {episode_reward}')
                break

            state, reward, done, info = env.step(chosen_action)
            episode_reward += reward

    env.close()
    return reward_history


#%% demo
if __name__ == '__main__':
    training_history = run(test=False)
    plt.plot(training_history)
    plt.xlabel('episode')
    plt.ylabel('epsiode reward')
    plt.show()
    run(test=True)

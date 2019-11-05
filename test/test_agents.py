import gym
import torch
import numpy as np
import wombat


env = gym.make('CartPole-v0')


def test_dqn():
    model = torch.nn.Sequential(
        torch.nn.Linear(env.observation_space.shape[0], 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, env.action_space.n))

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    def optimize(observation, action, target_reward):
        prediction = model(torch.tensor(observation, dtype=torch.float32))[action]
        loss = (prediction - target_reward) ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    agent = wombat.agents.DQN(lambda obs: model(torch.tensor(obs, dtype=torch.float32)).data.numpy(), optimize)
    episodes = wombat.run(agent=agent, environment=env, num_episodes=256, per_episode=wombat.train.offline())
    assert np.mean([episode.total_reward() for episode in episodes[-64:]]) > 100


def test_random_discrete():
    agent = wombat.agents.RandomDiscrete(env.action_space.n)
    episodes = wombat.run(agent=agent, environment=env, num_episodes=256)
    assert np.mean(list(episode.total_reward() for episode in episodes)) < 25

# wombat
Minimalist, flexible Python framework for reinforcement learning, especially experimental. Optimized for flexibility and ease of use.

## simplicity
Here's all the code you need to run the simplest experiment:
```python
import gym # OpenAI gym supported out of the box
import wombat
env = gym.make('CartPole-v0')
agent = wombat.agents.RandomDiscrete(env.action_space.n)
wombat.run(agent, env, num_episodes=4, per_step=env.render)
```

## tweakability
Want to write your own agent? No problem!
```python
class CyclicAgent: # no base class needed
   def __init__(self, num_possible_actions):
      self.num_possible_actions = num_possible_actions
   def act(self, episode): # should return selected action
      return len(episode) % self.num_possible_actions
   def train(self, steps): # should return training loss (eg. for prioritized experience replay)
      return 0

cyclic_agent = CyclicAgent(env.action_space.n)
wombat.run(cyclic_agent, env, num_episodes=4, per_step=env.render)
```
Want to manage the steps yourself, while retaining compatibility with wombat?
```python
episode = wombat.Episode() # will record all steps
for step in episode.run(agent, env):
   print(f'Action {step.action} resulted in reward of {step.reward}')
print(f'Episode finished, total reward: {episode.total_reward()}')
agent.train(episode.steps) # episode can be used with wombat, just like that
```

## quick code links
Full demo of training an agent: [demo.py](../blob/master/demo.py)  
Implementation of DQN agent: [dqn.py](../blob/master/wombat/agents/dqn.py)

## requirements
`tqdm` - loading bars  
`numpy` - utilities  
Technically speaking, that's it - but `gym` and `tensorflow` are both required to run the demo.

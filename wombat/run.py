from tqdm import trange
from wombat.episode import Episode
from wombat.utils import careful_call


def run(agent, environment, num_episodes, per_step=None, per_episode=None):
    '''
    Run agent on given OpenAI-gym-like environment
    Return a list of replays for all episodes
    '''
    episodes = []
    for episode_id in trange(num_episodes):
        episode = Episode()
        episodes.append(episode)
        for step in episode.run(agent=agent, environment=environment):
            careful_call(per_step, agent=agent, episodes=episodes)
        careful_call(per_episode, agent=agent, episodes=episodes)
    environment.close()
    return episodes

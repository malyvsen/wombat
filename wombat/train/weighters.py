import numpy as np


def uniform():
    '''Uniform weight for all episodes'''
    def result(episodes):
        return [1 for episode in episodes]
    return result


def prioritized(temperature=1, max_span=1):
    '''Softmax weight based on training loss'''
    def result(episodes):
        losses = np.array([episode.loss if hasattr(episode, 'loss') else 0 for episode in episodes], dtype=np.float32)
        losses -= np.mean(losses)
        span = np.max(losses) - np.min(losses)
        if span > max_span:
            losses = losses / span
        return np.exp(losses / temperature)
    return result

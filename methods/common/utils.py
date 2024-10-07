import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        """

        :param capacity:
        """
        self.capacity = capacity
        self.buffer = deque([], maxlen=capacity)

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self,batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states)
        )

    def __len__(self):
        return len(self.buffer)
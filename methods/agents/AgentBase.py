from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select_action(self, state=None):
        pass

    def store_transition(self, state=None, action=None, reward=None, next_state=None):
        pass

    def update(self):
        pass


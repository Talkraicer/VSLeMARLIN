# from AgentBase import Agent
from methods.agents.AgentBase import Agent

class RandomAgent(Agent):
    def __init__(self, **kwargs):
        super(RandomAgent, self).__init__()
        self.action_space = kwargs["action_space"]

    def select_action(self, state=None):
        action = self.action_space.sample()
        return action




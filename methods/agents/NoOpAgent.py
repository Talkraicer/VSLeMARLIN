from methods.agents.AgentBase import Agent

class NoOpAgent(Agent):
    def __init__(self, **kwargs):
        super(NoOpAgent,self).__init__()
        self.action_space = kwargs["action_space"]

    def select_action(self, state=None):
        action = self.action_space.sample()
        action.update({key: 1 for key in action})
        return action
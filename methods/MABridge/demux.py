from methods.agents.AgentBase import Agent
from methods.agents.RandomAgent import RandomAgent
from concurrent.futures import ThreadPoolExecutor, as_completed


class Demux(Agent):
    # def __init__(self, partitions=None, agent=RandomAgent, args=None):
    def __init__(self, partitions=None, agent=RandomAgent, mt=False, **kwargs):
        super(Demux, self).__init__()
        self.partitions = partitions
        self.mt = mt
        first_segment = next(iter(kwargs["action_space"]))
        self.action_space = kwargs["action_space"][first_segment]
        self.agents = {}
        for segs in partitions:
            self.agents[segs] = agent(action_space=self.action_space, action_size=kwargs["action_size"],
                                      state_size=kwargs["state_size"], replay_buffer_size=kwargs["replay_buffer_size"])

    def _select_action_worker(self, par, state):
        action = self.agents[par].select_action(state)
        return (par, action)

    def select_action(self, state=None):
        action = {}
        if self.mt:
            with ThreadPoolExecutor(max_workers=30) as executor:
                futures = [executor.submit(self._select_action_worker, par, state[par]) for par in self.partitions]

                for future in as_completed(futures):
                    result = future.result()
                    action[result[0]] = result[1]
        else:
            for seg, agent in self.agents.items():
                action[seg] = agent.select_action(state=state[seg])
        return action

    def _store_transition_worker(self, state, action, reward, next_state, par):
        self.agents[par].store_transition(state, action, reward, next_state)

    def store_transition(self, state=None, action=None, reward=None, next_state=None):
        if self.mt:
            with ThreadPoolExecutor(max_workers=30) as executor:
                futures = [executor.submit(self._store_transition_worker, state[par], action[par], reward[par],
                                           next_state[par], par) for par in self.partitions]
                for _ in as_completed(futures):
                    pass
        else:
            for par in self.partitions:
                self.agents[par].store_transition(state[par], action[par], reward[par], next_state[par])

    def _update_worker(self, par, n_times):
        for _ in n_times:
            self.agents[par].update()

    def update(self, n_times=1):
        if self.mt:
            with ThreadPoolExecutor(max_workers=30) as executor:
                futures = [executor.submit(self._update_worker, par, n_times) for par in self.partitions]
                for _ in as_completed(futures):
                    pass
        else:
            for par in self.partitions:
                self.agents[par].update()

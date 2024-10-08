from typing import SupportsFloat, Any

import gymnasium as gym
import numpy as np
from collections import OrderedDict
from env.SUMOAdpater import SUMOAdapter


class VSLenv(gym.Env):
    def __init__(self, config):
        self.config = config.get_config
        self.observation_space = gym.spaces.Dict(self._set_observations())
        self.observation_space.spaces = OrderedDict(self.observation_space.spaces)
        action_space = {seg: gym.spaces.Discrete(self.config["num_actions"]) for seg in self.config["segments"]}
        self.action_space = gym.spaces.Dict(action_space)
        self.action_space.spaces = OrderedDict(self.action_space.spaces)
        self.action_mapping = [-1, 0, 1] # 0: -1, 1:0, 2:+1

        self.state = {}
        self.timestep = 0
        self.time_since_change = {}
        self.last_speeds = {}
        self.demand = None
        self.seed = self.config["seed"]
        np.random.seed(self.seed)
        self.SUMO = SUMOAdapter(segments=self.config["segments"], gui=self.config["gui"], )

    def step(self, actions):
        for seg in self.config["segments"]:
            self.last_speeds[seg] = self._clamp((self.last_speeds[seg] + self.action_mapping[actions[seg]] *
                                     self.config["speed_step"]) , self.config["min_speed"], self.config["max_speed"])
            # self.time_since_change[seg] = ((1 - np.abs(self.action_mapping[actions[seg]]))
            self.time_since_change[seg] = (np.abs(self.action_mapping[actions[seg]])
                                           * (self.time_since_change[seg] + self.config["act_rate"]))
        # send actions to sumo
        self.SUMO.set_speed_limits(self.last_speeds)

        # run sumo for act_rate seconds
        rewards = self.SUMO.run_for_timesteps(self.config["act_rate"])
        self.timestep += self.config["act_rate"]

        # update state from SUMO
        self.state = self.SUMO.get_state()
        for seg in self.config["segments"]:
            self.state[seg].append(np.float32(self.last_speeds[seg]))
            self.state[seg].append(self.time_since_change[seg])

        done = self.SUMO.isFinish()

        # TODO: change the zero to the total reward (sum)
        total_reward = 0
        for seg in self.config["segments"]:
            total_reward += rewards[seg]
        return self.state, total_reward, done, False, rewards

    def reset(self, seed=None, demand="Low"):
        # check if a SUMO instance is already running
        try:
            self.SUMO.close()
        except:
            pass

        # reset the environment
        self.timestep = 0
        self.time_since_change = {seg: 0 for seg in self.config["segments"]}
        self.state = {seg: np.zeros(4) for seg in self.config["segments"]}
        self.last_speeds = {seg: self.config["nominal_speed"] for seg in self.config["segments"]}

        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
        if demand not in ["low", "medium", "high"]:
            raise ValueError("invalid demand profile value")
        self.demand = demand

        self.SUMO.init_simulation(seed, demand, output_folder=self.config["output_folder"],
                                  output_file=self.config["output_file"])

        # run sumo for warp-up
        for decisions in range(self.config["min_change_act"]):
            actions = {seg: 0 for seg in self.config["segments"]}    #default actions - do nothing
            self.step(actions)

        return self.state, {}

    def render(self):
        pass

    def close(self):
        self.SUMO.close()

    def _set_observations(self):
        return OrderedDict(
            {seg: gym.spaces.Tuple((
                gym.spaces.discrete.Discrete(self.config["max_num_vehs_seg"]),  # num_vehs
                gym.spaces.Box(self.config["min_speed"], self.config["max_speed"], shape=(), dtype=np.float32),  # mean_speed
                gym.spaces.Box(self.config["min_speed"], self.config["max_speed"], shape=(), dtype=np.float32),  # current_SL
                gym.spaces.Discrete(self.config["sim_duration"])  # time_passed
            ))
                for seg in self.config["segments"]})

    def _clamp(self, value, min_value, max_value):
        return max(min_value, min(value, max_value))
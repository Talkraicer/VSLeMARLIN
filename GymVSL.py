import gymnasium as gym
import numpy as np
from tqdm import tqdm
import wandb
from collections import OrderedDict

import traci
from stable_baselines3.common.env_checker import check_env
from settings import *
from log_utils import init_wandb_logger
import warnings
from run_utils import set_speed_limits
from cfg_files_VSL.create_cfg_files import set_cfg_file
from rou_files_VSL.create_rou_files import set_rou_file
from main import init_simulation
warnings.filterwarnings("ignore")
np.random.seed(SEED)


# List of relevant studied features

def set_observations(obs_type="default"):
    if obs_type == "default":
        obs_types = ["num_vehs", "mean_speed", "current_SL","time_passed"]
    else:
        raise ValueError("Invalid observation type")
    return OrderedDict(
        {seg: gym.spaces.Box(low=0, high=1, shape=(len(obs_types),), dtype=np.float32) for seg in SEGMENTS})



def extract_features_segment(segment, env):
    edges = segment.split("+")
    num_vehs = 0
    total_speed = 0
    for edge in edges:
        vehs = traci.edge.getLastStepVehicleIDs(edge)
        num_vehs += len(vehs)
        speeds = [traci.vehicle.getSpeed(veh) for veh in vehs]
        total_speed += sum(speeds)
    mean_speed = total_speed / num_vehs if num_vehs else 0

    # normalize the features
    num_vehs = num_vehs / env.config.max_num_vehs_seg
    mean_speed = mean_speed / env.config.max_speed
    return [num_vehs, mean_speed]



def action_wrapper(env, seg_actions):
    # set the speed limits according to the action
    for seg in SEGMENTS:
        if seg_actions[seg] == 0:
            env.last_speeds[seg] = max(env.config.max_speed, env.last_speeds[seg] - env.config.speed_step)
            env.time_since_change[seg] = env.config.act_rate
        elif seg_actions[seg] == 1:
            env.time_since_change[seg] += env.config.act_rate
        elif seg_actions[seg] == 2:
            env.last_speeds[seg] = min(env.config.max_speed, env.last_speeds[seg] + env.config.speed_step)
            env.time_since_change[seg] = env.config.act_rate
    set_speed_limits(env.last_speeds)

    # initialize the reward and last step vehicles
    rewards = {seg: 0 for seg in SEGMENTS}
    last_step_vehs = {seg: set() for seg in SEGMENTS}
    for seg in SEGMENTS:
        for edge in seg.split("+"):
            last_step_vehs[seg].union(set(traci.edge.getLastStepVehicleIDs(edge)))

    # run the step action_rate times
    for i in range(env.config.act_rate):
        traci.simulationStep(env.timestep)
        env.timestep += 1

        # calculate the reward
        curr_step_vehs = {seg: set() for seg in SEGMENTS}
        for seg in SEGMENTS:
            for edge in seg.split("+"):
                curr_step_vehs[seg].union(set(traci.edge.getLastStepVehicleIDs(edge)))
            rewards[seg] += len(last_step_vehs[seg] - curr_step_vehs[seg])
            last_step_vehs[seg] = curr_step_vehs[seg]

    # update the state
    env.state = {seg: extract_features_segment(seg, env) for seg in SEGMENTS}
    for seg in SEGMENTS:
        env.state[seg].append(env.last_speeds[seg] / env.config.max_speed)
        env.state[seg].append(env.time_since_change[seg]/env.config.sim_duration)

    env.state = {seg: np.array(env.state[seg]).astype(np.float32) for seg in SEGMENTS}

    done = traci.simulation.getMinExpectedNumber() <= 0

    return rewards, done


class Config:
    def __init__(self, train=True):
        # initialize the config with the default values
        self.act_rate = 15  # run each chosen SL for 15 seconds
        self.min_change_act = 4  # minimum number of actions before choosing action 0 or 2
        self.obs_type = "default"  # default observation space, as described above
        self.log_wandb = True  # log the results to wandb
        self.num_actions = 3  # number of actions - 0 decrease, 1 keep, 2 increase
        self.warmup = 40  # warmup num acts before starting the simulation

        # Speed Limits
        self.min_speed = 0
        self.max_speed = 37
        self.speed_step = 3
        self.max_num_vehs_seg = 1000 # TODO: change
        self.sim_duration = 3600 * 2 # 2 hours TODO: change


class SegVSL(gym.Env):
    def __init__(self, policy_name, config):
        self.config = config
        self.observation_space = gym.spaces.Dict(set_observations(config.obs_type))
        self.observation_space.spaces = OrderedDict(self.observation_space.spaces)
        action_space = {seg: gym.spaces.Discrete(config.num_actions) for seg in SEGMENTS}
        self.action_space = gym.spaces.Dict(action_space)
        self.action_space.spaces = OrderedDict(self.action_space.spaces)

        self.policy_name = policy_name
        self.state = {}
        self.timestep = 0
        self.time_since_change = {}
        self.last_speeds = {}


        if config.log_wandb:
            init_wandb_logger(f"RL_{EXP_NAME}", policy_name, delete_older=True)

    def observation(self):
        return self.state
    def reset(self, demand=None, seed=None):
        # check if a traci instance is already running
        try:
            traci.close()
        except:
            pass
        # reset the environment
        self.timestep = 0
        self.time_since_change = {seg: 0 for seg in SEGMENTS}
        self.state = {seg: np.zeros(4) for seg in SEGMENTS}
        self.last_speeds = {seg: 25 for seg in SEGMENTS}

        # randomize the demand and seed if not provided
        demand = DEMANDS[np.random.randint(0,3)] if demand is None else demand
        seed = np.random.randint(0, 10000) if seed is None else seed

        # set the rou and cfg files
        set_rou_file(demand, seed)
        set_cfg_file(demand, seed)
        sumoCfg = f"cfg_files_VSL/{demand}/{seed}/{EXP_NAME}.sumocfg"

        # initialize the simulation
        init_simulation((self.policy_name, sumoCfg))

        for i in range(self.config.warmup):
            action_wrapper(self, {seg: 1 for seg in SEGMENTS})
        return self.observation(), {}

    def step(self, seg_actions):
        rewards, done = action_wrapper(self, seg_actions)
        if self.config.log_wandb:
            log_msg = {}
            log_msg["Num DownSpeeds"] = sum([1 for seg in SEGMENTS if seg_actions[seg] == 0])
            log_msg["Num UpSpeeds"] = sum([1 for seg in SEGMENTS if seg_actions[seg] == 2])
            log_msg["Num SameSpeeds"] = sum([1 for seg in SEGMENTS if seg_actions[seg] == 1])
            log_msg["Total Reward"] = sum(rewards.values())
            wandb.log(log_msg)
        obs = self.observation()
        return obs, sum(rewards.values()), done, False, rewards


if __name__ == '__main__':
    # initialize the environment
    config = Config()
    config.log_wandb = False
    env = SegVSL("eMARLIN", config)
    check_env(env, warn=True)



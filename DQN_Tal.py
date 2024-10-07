# Based on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import torch
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from GymVSL import *

BATCH_SIZE = 8
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
NUM_EPISODES = 100
MEMORY_SIZE = 10000
# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def select_action(state, steps_done, policy_net, action_space):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[action_space.sample()]], device=device, dtype=torch.long)


def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def main():
    policy_name = "DQN_distri"
    env = SegVSL(policy_name, Config())
    logger = Logger(policy_name)
    agents = {}
    for seg in SEGMENTS:
        # Get number of actions from gym action space
        n_actions = env.action_space[seg].n
        # Get the number of state observations
        n_observations = len(env.observation_space[seg])

        policy_net = DQN(n_observations, n_actions).to(device)
        target_net = DQN(n_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(MEMORY_SIZE)

        agents[seg] = (policy_net, target_net, optimizer, memory)

    # training loop - train all of the agents at the same time on the same environment
    steps_done = 0
    mean_delays = {demand: [] for demand in DEMANDS}
    for i_episode in range(NUM_EPISODES):
        # Initialize the environment and state
        state, info = env.reset()
        total_reward = 0
        for t in count():
            # Select and perform an action
            actions = {}
            for seg in SEGMENTS:
                state_dict = state[seg]
                state_tensor = torch.tensor([state_dict], device=device, dtype=torch.float32)
                actions[seg] = select_action(state_tensor, steps_done, agents[seg][0], env.action_space[seg])
            next_state, mean_delay, done, _, agents_rewards = env.step(actions)
            total_reward += sum(agents_rewards.values())
            logger.log(actions, agents_rewards)
            # Store the transition in memory
            for seg in SEGMENTS:
                state_dict = state[seg]
                next_state_dict = next_state[seg]
                agents[seg][3].push(
                    torch.tensor([state_dict], device=device, dtype=torch.float32),
                    actions[seg],
                    torch.tensor([next_state_dict], device=device, dtype=torch.float32),
                    torch.tensor([agents_rewards[seg]], device=device, dtype=torch.float32)
                )

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            for seg in SEGMENTS:
                optimize_model(agents[seg][3], agents[seg][0], agents[seg][1], agents[seg][2])

            steps_done += 1
            if done:
                break
        print("*" * 50)
        print(f"Episode {i_episode} finished after {t + 1} timesteps, total reward: {total_reward}")
        print("Demand:", env.demand)
        print("seed:", env.seed)
        print("Mean delay:", mean_delay)
        print("*" * 50)
        mean_delays[env.demand].append(mean_delay)
    # end wandb
    logger.close()
    env.close()
    with open("delays.txt", "w") as f:
        for demand in mean_delays:
            f.write(f"Demand: {demand}\n")
            f.write(f"Mean delays: {mean_delays[demand]}\n")
            f.write("*" * 50 + "\n")

if __name__ == '__main__':
    main()

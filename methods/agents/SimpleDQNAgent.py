from methods.common.models import FFModel
from methods.common.utils import ReplayBuffer

import random

import torch
import torch.optim as optim

class DQNAgent:
    def __init__(self, state_size, action_size, replay_buffer_size, gamma=0.99, lr=1e-3, batch_size=64,
                 target_update_freq=1000):
        """
        Simple flat FF DQN agent
        :param state_size (int): Dimension of the state space.
        :param action_size (int): Number of possible actions.
        :param replay_buffer (ReplayBuffer): The experience replay buffer capacity.
        :param gamma (float): Discount factor.
        :param lr (float): Learning rate for the optimizer.
        :param batch_size (int): Number of experiences to sample per batch.
        :param target_update_freq (int): How often to update the target network.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.q_network = FFModel(state_size, action_size)
        self.target_q_network = FFModel(state_size, action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.steps_done = 0

    def select_action(self, state, epsilon=0.1):
        """
        Epsilon-greedy action selection.
        :param state (ndarray): Current state of the environment.
        :param epsilon (float): Probability of choosing a random action (exploration).
        :return: int: Chosen action.
        """
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state):
        self.replay_buffer.add(state, action, reward, next_state)

    def update(self):
        """
        :return:
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)

        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_q_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q_values

        loss = torch.nn.MSELoss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.steps_done += 1


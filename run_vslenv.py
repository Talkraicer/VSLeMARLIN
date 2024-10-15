from env.config import Config
from env.VSLenv import VSLenv
from methods.agents.SimpleDQNAgent import DQNAgent
from methods.MABridge.demux import Demux
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=100, help='number of episodes for training')
parser.add_argument('--steps', type=int, default=100, help='number of steps per episode')
parser.add_argument('--n_updates', type=int, default=1, help='number gradient updates per episode')
args = parser.parse_args()


def main():
    segs = [f'Fi{i}' for i in range(1, 6)] + \
            [f'Mo{i}' for i in range(1, 6)] + \
            [f'Mi{i}+S{i}' for i in range(1, 6)] + \
            [f'E{i}' for i in range(1, 6)]

    seed = 42
    demand = "low"
    cfg = Config(segments=segs, gui=False)
    env = VSLenv(config=cfg)
    policy = Demux(mt=True, partitions=segs, action_space=env.action_space, agent=DQNAgent, state_size=4, action_size=3, replay_buffer_size=10000)
    # policy = RandomAgent(action_space=env.action_space)
    # policy = NoOpAgent(action_space=env.action_space)

    for eps in range(args.episodes):
        total_reward = 0
        state, _ = env.reset(seed=seed, demand=demand)
        for steps in range(args.steps):
            action = policy.select_action(state)
            next_state, scalar_reward, _, _, rewards = env.step(action)
            policy.store_transition(state, action, rewards, next_state)
            state = next_state
            total_reward += scalar_reward
        print(f"episode {eps} ended with reward {total_reward:.2f}")
        policy.update(args.n_updates)
    env.close()


if __name__ == "__main__":
    main()
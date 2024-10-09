from env.config import Config
from env.VSLenv import VSLenv
import sys

def main():
    segs = [f'Fi{i}' for i in range(1, 6)] + \
            [f'Mo{i}' for i in range(1, 6)] + \
            [f'Mi{i}+S{i}' for i in range(1, 6)] + \
            [f'E{i}' for i in range(1, 6)]

    seed = 42
    demand = "low"
    cfg = Config(segments=segs, gui=False)
    env = VSLenv(config=cfg)

    total_reward = 0
    state, _ = env.reset(seed=seed, demand=demand)
    for steps in range(60):
        action = env.action_space.sample()
        state, _, _, _, rewards = env.step(action)
        print(steps, state)

    env.close()


if __name__ == "__main__":
    main()
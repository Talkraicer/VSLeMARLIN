from env.config import Config
from env.VSLenv import VSLenv

def main():
    segs = [f'Fi{i}' for i in range(1, 6)] + \
            [f'Mo{i}' for i in range(1, 6)] + \
            [f'Mi{i}+S{i}' for i in range(1, 6)] + \
            [f'E{i}' for i in range(1, 6)]

    seed = 42
    demand = "low"
    cfg = Config(segments=segs)
    env = VSLenv(config=cfg)

    total_reward = 0
    state, _ = env.reset(seed=seed, demand=demand)
    action = env.action_space.sample()
    state, _, _, _, rewards = env.step(action)
    print ("hello")


if __name__ == "__main__":
    main()
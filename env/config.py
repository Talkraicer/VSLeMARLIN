class Config:
    def __init__(self, act_rate = 15, min_change_act = 4, obs_type = "default", log_wandb = True, num_actions=3,
                 warmup = 40, min_speed = 0, max_speed = 120, nominal_speed = 80, speed_step = 5, max_num_vehs_seg = 1000,
                 sim_duration = 3600 * 2, seed=42, segments=None):
        # initialize the config with the default values
        self._args = dict()
        self._args["act_rate"] = act_rate # run each decision step for 15 seconds
        self._args["min_change_act"] = min_change_act # minimum number of actions before choosing action 0 or 2
        self._args["obs_type"] = obs_type # default observation space, as described above
        self._args["log_wandb"] = log_wandb # log the results to wandb
        self._args["num_actions"] = num_actions  # number of actions - 0 decrease, 1 keep, 2 increase
        self._args["warmup"] = warmup # warmup num acts before starting the simulation

        # Speed Limits
        self._args["min_speed"] = min_speed
        self._args["max_speed"] = max_speed
        self._args["speed_step"] = speed_step
        self._args["nominal_speed"] = nominal_speed # the nominal speed of the simulation
        self._args["max_num_vehs_seg"] = max_num_vehs_seg # TODO: change
        self._args["sim_duration"] = sim_duration # 2 hours TODO: change

        self._args["segments"] = segments
        self._args["seed"] = seed


    @property
    def get_config(self):
        return self._args

from abc import ABC, abstractmethod
import wandb

class LoggerBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def log(self, message):
        pass

class WandbLogger(LoggerBase):
    def __init__(self, policy_name, exp_name, delete=True, segments=None):
        super(WandbLogger, self).__init__()
        self.policy_name = policy_name
        self.segments = segments
        self._init_wandb_logger(f"RL_{exp_name}", self.policy_name, delete_older=delete)

    def log(self, message):
        seg_actions, rewards = message["seg_actions"], message["rewards"]
        log_msg = {}
        log_msg["Num DownSpeeds"] = sum([1 for seg in self.segments if seg_actions[seg] == 0])
        log_msg["Num UpSpeeds"] = sum([1 for seg in self.segments if seg_actions[seg] == 2])
        log_msg["Num SameSpeeds"] = sum([1 for seg in self.segments if seg_actions[seg] == 1])
        log_msg["Total Reward"] = sum(rewards.values())
        wandb.log(log_msg)

    def _init_wandb_logger(self, proj_name, policy_name, delete_older=False):
        api = wandb.Api()
        username = api.default_entity

        projects = api.projects(username)
        if delete_older and proj_name in [proj.name for proj in projects]:
            # Retrieve the run ID (you can also manually set this if you know the ID)
            runs = api.runs(f"{username}/{proj_name}")

            # Delete the run if it exists
            deleted = False
            for run in runs:
                if run.name == policy_name:
                    run = api.run(f"{username}/{proj_name}/{run.id}")
                    run.delete()
                    deleted = True
                    break
            if not deleted:
                print(f"Run {policy_name} not found")
        wandb.init(project=proj_name, name=policy_name)

class CSVLogger(LoggerBase):
    def __init__(self):
        super(CSVLogger, self).__init__()

    def log(self, message):
        pass

class TBLogger(LoggerBase):
    def __init__(self):
        super(TBLogger, self).__init__()

    def log(self, message):
        pass



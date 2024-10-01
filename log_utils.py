import wandb
from settings import *


def init_wandb_logger(proj_name, policy_name, delete_older=False):
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

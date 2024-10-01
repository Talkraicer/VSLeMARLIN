from xml.etree import ElementTree as ET
import pandas as pd
import numpy as np
import wandb
import os
import traci
from results_utils import output_file_to_df, calc_stats_metric
from utils import exp_name
results_reps_folder = "results_reps"
NUM_EDGES = 8

def init_wandb_logger(policy_name,av_rate,delete_older=False):
    proj_name = exp_name + "_" + str(av_rate)

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

import os
import traci
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import wandb
import time
from log_utils import init_wandb_logger
import pickle



def extract_sim_features():
    # calc all vehicles speed in the road
    vehIDs = traci.vehicle.getIDList()
    mean_speed = np.mean([traci.vehicle.getSpeed(vehID) for vehID in vehIDs])
    mean_speed_in_end_PTL = traci.lane.getLastStepMeanSpeed("E6_3")
    num_vehs_in_PTL = sum(
        [traci.lane.getLastStepVehicleNumber(l) for l in traci.lane.getIDList() if len(traci.lane.getAllowed(l)) > 0])
    num_total_vehs = len(vehIDs)
    num_hdv_in_end_PTL = sum([traci.lane.getLastStepVehicleNumber(f"E6_{i}") for i in range(3)])

    num_allowed_vehs_PTL = sum([1 for vehID in vehIDs if traci.vehicle.getVehicleClass(vehID) in ["private", "bus"]])

    PTL_speeds = []
    for e_idx in range(NUM_EDGES):
        edge = "E" + str(e_idx)
        if 1 <= e_idx < NUM_EDGES - 1:
            num_lanes = traci.edge.getLaneNumber(edge)
            PTL_idx = num_lanes - 1
            PTL_speeds += [traci.vehicle.getSpeed(vehID) for vehID in
                           traci.lane.getLastStepVehicleIDs(f"{edge}_{PTL_idx}")]

    mean_speed_in_PTL = np.mean(PTL_speeds)

    # calc arrived passengers mean total delay
    output_file = f"{results_reps_folder}/{output_file}"
    with open(output_file, "r") as f:
        txt = f.read()
    start_arriving = "<tripinfo " in txt
    mean_pass_delay = 0
    if start_arriving:
        # fix end of <tripinfo> tag
        with open(output_file, "a+") as f:
            f.write("</tripinfos>")
        df = output_file_to_df(output_file)
        total_delay = calc_stats_metric(df, "totalDelay", diff=False)
        mean_pass_delay = total_delay.loc["avg_totalDelay", "Passenger"]

        df_timestamp = df[df["arrivalTime"] > t - log_rate]
        total_delay_timestamp = calc_stats_metric(df_timestamp, "totalDelay", diff=False)
        mean_pass_delay_timestamp = total_delay_timestamp.loc["avg_totalDelay", "Passenger"]

        # remove the <tripinfo> tag
        with open(output_file, "r") as f:
            lines = f.readlines()
        with open(output_file, "w") as f:
            f.writelines(lines[:-1])

        log_msg = {"num_vehs_in_PTL": num_vehs_in_PTL, "num_total_vehs": num_total_vehs,
                   "num_hdv_in_end_PTL": num_hdv_in_end_PTL, "mean_speed": mean_speed,
                   "mean_speed_in_end_PTL": mean_speed_in_end_PTL, "mean_pass_delay": mean_pass_delay,
                   "num_allowed_vehs_PTL": num_allowed_vehs_PTL, "mean_speed_in_PTL": mean_speed_in_PTL,
                   "mean_pass_delay_timestamp": mean_pass_delay_timestamp
                   }
        return log_msg

def set_speed_limits(segments_speeds):
    for seg,speed in segments_speeds.items():
        edges = seg.split("+")
        for edge in edges:
            traci.edge.setMaxSpeed(edge, speed)

def handle_step(t, policy_name, demand=None, seed=None, log_rate=100):
    if log_rate and t % log_rate == 0:
        if not demand or not seed:
            raise ValueError("demand and seed must be provided if log_rate is not None")
        if t == 0:
            init_wandb_logger(demand+"_"+seed, policy_name, delete_older=True)


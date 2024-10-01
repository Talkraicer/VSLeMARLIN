import os
import sys

from tqdm import tqdm
from multiprocessing import Pool

from run_utils import *
from settings import *
import traci

POLICIES = ["Nothing"]

if GUI:
    NUM_PROCESSES = 1

if 'SUMO_HOME' in os.environ:
    sumo_path = os.environ['SUMO_HOME']
    sys.path.append(os.path.join(sumo_path, 'tools'))
    # check operational system - if it is windows, use sumo.exe if linux, use sumo
    if os.name == 'nt':
        sumoBinary = os.path.join(sumo_path, 'bin', 'sumo-gui.exe') if GUI else \
            os.path.join(sumo_path, 'bin', 'sumo.exe')
    else:
        sumoBinary = os.path.join(sumo_path, 'bin', 'sumo-gui') if GUI else \
            os.path.join(sumo_path, 'bin', 'sumo')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def init_simulation(arg):
    policy_name, sumoCfg = arg
    seed = sumoCfg.split("/")[-2]
    demand = sumoCfg.split("/")[-3]
    sumoCmd = [sumoBinary, "-c", sumoCfg, "--tripinfo-output"]
    exp_output_name = f"{RESULTS_FOLDER}/results_reps/{demand}/{seed}/"
    os.makedirs(exp_output_name, exist_ok=True)
    exp_output_name += policy_name + "_" + EXP_NAME + ".xml"
    sumoCmd.append(exp_output_name)
    traci.start(sumoCmd)
    return policy_name, sumoCfg, demand, seed


def simulate(arg, log_wandb=False):
    policy_name, sumoCfg, demand, seed = init_simulation(arg)
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        if log_wandb:
            handle_step(step, policy_name, demand, seed, log_rate=0)
        else:
            handle_step(step, policy_name, demand, seed)
        traci.simulationStep(step)
        step += 1
    traci.close()
    # wandb.finish()


def parallel_simulation(args):
    with Pool(NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(simulate, args), total=len(args)))


def simulate_policies(sumoCfgPaths):
    args = []
    for policy in POLICIES:
        for sumoCfg in sumoCfgPaths:
            args.append((policy, sumoCfg))
    parallel_simulation(args)


def run_random_experiments():
    sumoCfgPaths = []
    for demand in DEMANDS:
        for seed in os.listdir(f"cfg_files_{EXP_NAME}/{demand}"):
            sumoCfgPaths.append(f"cfg_files_{EXP_NAME}/{demand}/{seed}/{EXP_NAME}.sumocfg")
    print("Number of sumoCfg files: ", len(sumoCfgPaths))
    if GUI:
        sumoCfgPaths = [sumoCfgPaths[3]]
    simulate_policies(sumoCfgPaths)


def run_experiment(demand, seed):
    sumoCfgPaths = [f"cfg_files_VSL/{demand}/{seed}/{EXP_NAME}.sumocfg"]
    simulate_policies(sumoCfgPaths)


def main():
    run_random_experiments()

if __name__ == "__main__":
    main()

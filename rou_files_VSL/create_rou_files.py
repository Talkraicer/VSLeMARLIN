import os
from xml.etree import ElementTree as ET
import copy
import pandas as pd
import matplotlib.pyplot as plt
from settings import *
import numpy as np

DEMAND_SIZES = {"low": 2500, "medium": 3500, "high": 5000}
NUM_LANES = 4
def create_vehicle_amounts(demand, plot = False):
    size = DEMAND_SIZES[demand]
    grow_rates = [1, 1.2, 1.4, 1.6, 1.8, 2.0]
    decay_rates = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0]
    # every key stands for 10 min
    veh_amounts = {i:size*grow_rate for i,grow_rate in enumerate(grow_rates)}
    veh_amounts.update({i+6:size*decay_rate for i,decay_rate in enumerate(decay_rates)})

    if plot:
        df = pd.DataFrame(veh_amounts.items(), columns = ['10min', 'Vehicles'])
        df.plot(x = '10min', y = 'Vehicles', kind = 'bar')
        plt.suptitle('Vehicles demand per hour')
        plt.grid()
        plt.savefig(f'{ROOT}/rou_files_{EXP_NAME}/{EXP_NAME}_{demand}.png')
        # plt.show()
        plt.close()
    return veh_amounts


def set_rou_file(demand,seed, HOUR_LEN = 600, plot = False):

    veh_amounts = create_vehicle_amounts(demand, plot = plot)
    in_junc = "J1"
    out_junc = "J7"
    in_ramps = [f'i{i}' for i in range(1,6)]
    out_ramps = [f'o{i}' for i in range(1,6)]

    # Load and parse the XML file
    tree = ET.parse(f'{ROOT}/rou_files_{EXP_NAME}/{EXP_NAME}.rou.xml')
    root = tree.getroot()

    np.random.seed(seed)
    in_probs = np.random.uniform(0,0.2,5)
    out_probs = np.random.uniform(0,0.2,5)
    for hour,hour_demand in veh_amounts.items():
        total_arrival_prob = hour_demand / 3600
        taken = 0
        for i, (in_ramp, in_prob) in enumerate(zip(in_ramps, in_probs)):
            left_in = 1
            # In ramps to Out ramps
            for j,(out_ramp, out_prob) in enumerate(zip(out_ramps, out_probs)):
                if i > j:
                    continue
                prob = in_prob * out_prob
                left_in -= out_prob
                flow_prob = total_arrival_prob * prob
                if total_arrival_prob * HOUR_LEN > 1:
                    flow = ET.Element('flow', id=f'flow_{hour}_{in_ramp}_{out_ramp}', type="DEFAULT_VEHTYPE", begin=str(hour * HOUR_LEN),
                                      fromJunction=in_ramp, toJunction=out_ramp,end=str((hour +1) * HOUR_LEN), probability=f"{flow_prob}",departSpeed="desired")
                    flow.tail = '\n\t'
                    root.append(flow)
                else:
                    print(f'hour {hour} in_ramp {in_ramp} out_ramp {out_ramp} prob {flow_prob}')
            # In ramps to Out junction
            flow = ET.Element('flow', id=f'flow_{hour}_{in_ramp}_{out_junc}', type="DEFAULT_VEHTYPE",
                              begin=str(hour * HOUR_LEN),
                              fromJunction=in_ramp, toJunction=out_junc, end=str((hour + 1) * HOUR_LEN),
                              probability=f"{left_in* in_prob*total_arrival_prob}",departSpeed="desired")
            flow.tail = '\n\t'
            root.append(flow)

        # In junction to Out ramps
        for out_ramp, out_prob in zip(out_ramps, out_probs):
            taken += out_prob
            for lane in range(NUM_LANES):
                flow_prob = total_arrival_prob * out_prob / NUM_LANES
                flow = ET.Element('flow', id=f'flow_{hour}_{in_junc}_{out_ramp}_{lane}', type="DEFAULT_VEHTYPE", departLane=str(lane),
                                  begin=str(hour * HOUR_LEN),
                                  fromJunction=in_junc, toJunction=out_ramp, end=str((hour + 1) * HOUR_LEN),
                                  probability=f"{flow_prob}",departSpeed="desired")
                flow.tail = '\n\t'
                root.append(flow)

        # In junction to Out junction
        in_out_prob = 1 - taken
        assert in_out_prob >= 0
        for lane in range(NUM_LANES):
            flow_prob = total_arrival_prob * in_out_prob / NUM_LANES
            flow = ET.Element('flow', id=f'flow_{hour}_{in_junc}_{out_junc}_{lane}', type="DEFAULT_VEHTYPE", departLane=str(lane),
                              begin=str(hour * HOUR_LEN),
                              fromJunction=in_junc, toJunction=out_junc, end=str((hour + 1) * HOUR_LEN),
                              probability=f"{flow_prob}",departSpeed="desired")
            flow.tail = '\n\t'
            root.append(flow)

    # Save the changes back to the file
    os.makedirs(f"{ROOT}/rou_files_{EXP_NAME}/{demand}/{seed}", exist_ok=True)
    tree.write(f'{ROOT}/rou_files_{EXP_NAME}/{demand}/{seed}/{EXP_NAME}.rou.xml')


if __name__ == '__main__':
    np.random.seed(SEED)
    seeds = [np.random.randint(0, 10000) for _ in range(10)]
    for demand in DEMANDS:
        for seed in seeds:
            set_rou_file(demand, seed, plot = True)
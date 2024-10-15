import traci
import numpy as np
import os
from xml.etree import ElementTree as ET

class SUMOAdapter():
    def __init__(self, segments, config_folder="SUMOconfig", lane_num=4, gui=False):
        curdir = os.path.dirname(os.path.abspath(__file__))
        self.config_folder = os.path.join(curdir, config_folder)
        self.route_template = "route_template.xml"
        self.config_template = "config_template.sumocfg"
        self.network_file = "network.xml"
        self.segments = segments
        self.gui = gui
        self.demand_map = {"low":2500, "medium": 3500, "high": 5000}
        self.lane_num = lane_num

    def set_speed_limits(self, speeds):
        for seg, speed in speeds.items():
            edges = seg.split("+")
            for edge in edges:
                traci.edge.setMaxSpeed(edge, speed)

    def get_vehicle_IDs(self):
        step_vehs = {seg: set() for seg in self.segments}
        for seg in self.segments:
            for edge in seg.split("+"):
                step_vehs[seg] = step_vehs[seg].union(set(traci.edge.getLastStepVehicleIDs(edge)))
        return step_vehs

    def run_for_timesteps(self, timesteps):
        route_completions = {seg: 0 for seg in self.segments}
        last_vehicles_IDs = self.get_vehicle_IDs()

        for i in range(timesteps):
            traci.simulationStep()

            curr_vehicles_IDs = self.get_vehicle_IDs()
            for seg in self.segments:
                route_completions[seg] += len(last_vehicles_IDs[seg] - curr_vehicles_IDs[seg])
            last_vehicles_IDs = curr_vehicles_IDs

        completion_rate = {seg: route_completions[seg] / timesteps for seg in self.segments}

        return completion_rate

    def get_state(self):
        states = dict()
        for seg in self.segments:
            edges = seg.split("+")
            num_vehs = 0
            total_speed = 0
            for edge in edges:
                vehs = traci.edge.getLastStepVehicleIDs(edge)
                num_vehs += len(vehs)
                speeds = [traci.vehicle.getSpeed(veh) for veh in vehs]
                total_speed += sum(speeds)
            mean_speed = total_speed / num_vehs if num_vehs else 0

            # normalize the features
            num_vehs = num_vehs
            mean_speed = mean_speed
            # convert mean_speed to dtype float32
            mean_speed = np.float32(mean_speed)
            states[seg] = [num_vehs, mean_speed]

        return states

    def isFinish(self):
        return traci.simulation.getMinExpectedNumber() <= 0

    def close(self):
        traci.close()

    def init_simulation(self, seed, demand, route_file="route.xml",
                        config_file="config.sumocfg", output_folder="experiments", output_file="output.xml"):
        demand = self.demand_map[demand]
        self.output_file = output_file
        output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", output_folder)
        self._create_route_file(demand, seed, route_file)
        self._create_config_file(seed, route_file=route_file, config_file=config_file)
        self._init_sumo(config_file, output_folder)

    def _create_route_file(self, demand, seed, route_file, hour_len = 600):
        # create a bell like demand shape
        grow_rates = [1, 1.2, 1.4, 1.6, 1.8, 2.0]
        decay_rates = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0]
        # every key stands for 10 min
        veh_amounts = {i: demand * grow_rate for i, grow_rate in enumerate(grow_rates)}
        veh_amounts.update({i + 6: demand * decay_rate for i, decay_rate in enumerate(decay_rates)})

        in_junc = "J1"
        out_junc = "J7"
        in_ramps = [f'i{i}' for i in range(1, 6)]
        out_ramps = [f'o{i}' for i in range(1, 6)]

        tree = ET.parse(os.path.join(self.config_folder, self.route_template))
        root = tree.getroot()
        np.random.seed(seed)
        in_probs = np.random.uniform(0, 0.2, 5)
        out_probs = np.random.uniform(0, 0.2, 5)
        for hour, hour_demand in veh_amounts.items():
            total_arrival_prob = hour_demand / 3600
            taken = 0
            for i, (in_ramp, in_prob) in enumerate(zip(in_ramps, in_probs)):
                left_in = 1
                # In ramps to Out ramps
                for j, (out_ramp, out_prob) in enumerate(zip(out_ramps, out_probs)):
                    if i > j:
                        continue
                    prob = in_prob * out_prob
                    left_in -= out_prob
                    flow_prob = total_arrival_prob * prob
                    if total_arrival_prob * hour_len > 1:
                        flow = ET.Element('flow', id=f'flow_{hour}_{in_ramp}_{out_ramp}', type="DEFAULT_VEHTYPE",
                                          begin=str(hour * hour_len),
                                          fromJunction=in_ramp, toJunction=out_ramp, end=str((hour + 1) * hour_len),
                                          probability=f"{flow_prob}", departSpeed="desired")
                        flow.tail = '\n\t'
                        root.append(flow)
                    else:
                        print(f'hour {hour} in_ramp {in_ramp} out_ramp {out_ramp} prob {flow_prob}')
                # In ramps to Out junction
                flow = ET.Element('flow', id=f'flow_{hour}_{in_ramp}_{out_junc}', type="DEFAULT_VEHTYPE",
                                  begin=str(hour * hour_len),
                                  fromJunction=in_ramp, toJunction=out_junc, end=str((hour + 1) * hour_len),
                                  probability=f"{left_in * in_prob * total_arrival_prob}", departSpeed="desired")
                flow.tail = '\n\t'
                root.append(flow)

            # In junction to Out ramps
            for out_ramp, out_prob in zip(out_ramps, out_probs):
                taken += out_prob
                for lane in range(self.lane_num):
                    flow_prob = total_arrival_prob * out_prob / self.lane_num
                    flow = ET.Element('flow', id=f'flow_{hour}_{in_junc}_{out_ramp}_{lane}', type="DEFAULT_VEHTYPE",
                                      departLane=str(lane),
                                      begin=str(hour * hour_len),
                                      fromJunction=in_junc, toJunction=out_ramp, end=str((hour + 1) * hour_len),
                                      probability=f"{flow_prob}", departSpeed="desired")
                    flow.tail = '\n\t'
                    root.append(flow)

            # In junction to Out junction
            in_out_prob = 1 - taken
            assert in_out_prob >= 0
            for lane in range(self.lane_num):
                flow_prob = total_arrival_prob * in_out_prob / self.lane_num
                flow = ET.Element('flow', id=f'flow_{hour}_{in_junc}_{out_junc}_{lane}', type="DEFAULT_VEHTYPE",
                                  departLane=str(lane),
                                  begin=str(hour * hour_len),
                                  fromJunction=in_junc, toJunction=out_junc, end=str((hour + 1) * hour_len),
                                  probability=f"{flow_prob}", departSpeed="desired")
                flow.tail = '\n\t'
                root.append(flow)

        # Save the changes back to the file
        # os.makedirs(f"{ROOT}/rou_files_{EXP_NAME}/{demand}/{seed}", exist_ok=True)
        output_file = os.path.join(self.config_folder, route_file)
        tree.write(output_file)
        # tree.write(f'{ROOT}/rou_files_{EXP_NAME}/{demand}/{seed}/{EXP_NAME}.rou.xml')

    def _create_config_file(self, seed, route_file, config_file):
        # Load and parse the XML file
        tree = ET.parse(os.path.join(self.config_folder, self.config_template))
        root = tree.getroot()

        # set route file
        route_file_pointer = root.find("input").find('route-files')
        route_file_pointer.set('value', os.path.join(self.config_folder, route_file))

        # set net file
        net_file_pointer = root.find("input").find('net-file')
        net_file_pointer.set('value', os.path.join(self.config_folder, self.network_file))

        # set seed
        seed_element_pointer = root.find("random_number").find('seed')
        seed_element_pointer.set('value', str(seed))

        output_config_file = os.path.join(self.config_folder, config_file)
        tree.write(output_config_file)

    def _init_sumo(self, config_file, results_folder=None):
        sumo_binary = self._get_sumo_entrypoint()
        cfg = os.path.join(self.config_folder, config_file)
        sumo_cmd = [sumo_binary, "-c", cfg]
        results_file = os.path.join(results_folder, self.output_file)
        sumo_cmd = sumo_cmd + ["--tripinfo-output", results_file]
        # print(sumo_cmd)
        traci.start(sumo_cmd)

    def _get_sumo_entrypoint(self):
        if 'SUMO_HOME' in os.environ:
            sumo_path = os.environ['SUMO_HOME']
            # check operational system - if it is windows, use sumo.exe if linux/macos, use sumo
            if os.name == 'nt':
                sumo_binary = os.path.join(sumo_path, 'bin', 'sumo-gui.exe') if self.gui else \
                    os.path.join(sumo_path, 'bin', 'sumo.exe')
            else:
                sumo_binary = os.path.join(sumo_path, 'bin', 'sumo-gui') if self.gui else \
                    os.path.join(sumo_path, 'bin', 'sumo')
        else:
            raise Exception("please declare environment variable 'SUMO_HOME'")
        return sumo_binary
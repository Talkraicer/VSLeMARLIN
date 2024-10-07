import traci
import numpy as np

class SUMOAdapter():
    def __init__(self, segments):
        self.segments = segments

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
        completion_rate = {seg: 0 for seg in self.segments}
        last_vehicles_IDs = self.get_vehicle_IDs(self.segments)

        for i in range(timesteps):
            traci.simulationStep()

            curr_vehicles_IDs = self.get_vehicle_IDs(self.segments)
            completion_rate += len(curr_vehicles_IDs - last_vehicles_IDs)
            last_vehicles_IDs = curr_vehicles_IDs

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

    def init_simulation(self):
        pass

    def _set_rou_file(self, demand, seed):
        pass

    def _set_cfg_file(self, demand, seed):
        pass

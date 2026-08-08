# envs/sumo_manager.py
import os
import sys
import traci
from sumolib import checkBinary

class SumoManager:
    def __init__(self, config_file, use_gui=False):
        """
        Initializes the SUMO TraCI connection.
        :param config_file: Path to your .sumocfg file in sumo_config/
        :param use_gui: True to see the cars moving (local), False for training (Colab)
        """
        # 1. Setup SUMO_HOME tools
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("Please declare environment variable 'SUMO_HOME'")

        # 2. Select binary (sumo or sumo-gui)
        self.sumo_binary = checkBinary('sumo-gui') if use_gui else checkBinary('sumo')
        self.config_file = config_file
        self.label = "sim1"

    def start(self):
        """Starts the simulation."""
        traci.start([self.sumo_binary, "-c", self.config_file], label=self.label)

    def step(self):
        """Advances the simulation by one time slot (Section III: discrete time slots)."""
        traci.simulationStep()

    def get_vehicle_data(self):
        """
        Extracts real-time data for all vehicles currently on the map.
        Ref: Section IV-B (Historical trajectory detection input)
        """
        vehicles = {}
        vehicle_ids = traci.vehicle.getIDList()
        
        for v_id in vehicle_ids:
            pos = traci.vehicle.getPosition(v_id)  # (x, y)
            speed = traci.vehicle.getSpeed(v_id)    # m/s
            
            # We store the data needed for the State Space (Eq. 24)
            vehicles[v_id] = {
                'pos': pos,
                'speed': speed
            }
        return vehicles

    def close(self):
        """Closes the connection."""
        traci.close()

# --- Quick Local Test Logic ---
if __name__ == "__main__":
    # Ensure you have a placeholder .sumocfg in your sumo_config folder to test this
    # manager = SumoManager("sumo_config/hangzhou.sumocfg", use_gui=True)
    # manager.start()
    # for _ in range(10):
    #     manager.step()
    #     print(manager.get_vehicle_data())
    # manager.close()
    print("SumoManager class defined successfully.")
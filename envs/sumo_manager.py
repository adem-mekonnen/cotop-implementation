import sys
import traci
from typing import List
from envs.entities import Vehicle

class SumoManager:
    """
    Bridge between the VEC environment and the SUMO traffic simulator.
    Uses TraCI to control the simulation and extract vehicle mobility data.
    """
    def __init__(self, sumocfg_path: str, use_gui: bool = False):
        self.sumocfg_path = sumocfg_path
        self.use_gui = use_gui
        # Automatically toggle between CLI and GUI mode
        self.sumo_binary = "sumo-gui" if self.use_gui else "sumo"

    def start_simulation(self):
        """Starts the SUMO simulation via TraCI."""
        sumo_cmd = [self.sumo_binary, "-c", self.sumocfg_path]
        try:
            traci.start(sumo_cmd)
            print(f"SUMO simulation started with {self.sumo_binary}")
        except Exception as e:
            print(f"Failed to start SUMO. Ensure it is installed and in your PATH. Error: {e}")
            sys.exit(1)

    def step(self):
        """Advances the SUMO simulation by one time step."""
        traci.simulationStep()

    def get_vehicle_data(self) -> List[Vehicle]:
        """
        Retrieves active vehicles from SUMO and formats them.
        
        Returns:
            List of Vehicle dataclass objects containing (x,y) positions and speeds.
        """
        vehicles = []
        vehicle_ids = traci.vehicle.getIDList()
        
        for v_id_str in vehicle_ids:
            # TraCI returns strings for vehicle IDs. We convert it to a numeric ID 
            # for our dataclass representation.
            try:
                numeric_id = int(''.join(filter(str.isdigit, v_id_str)))
            except ValueError:
                numeric_id = hash(v_id_str) % 1000000
                
            pos = traci.vehicle.getPosition(v_id_str)
            speed = traci.vehicle.getSpeed(v_id_str)
            
            # The dwell_time_T_stay requires knowledge of RSU coverage radius 
            # which is typically calculated externally in the environment wrapper.
            # We initialize it as 0.0 here.
            dwell_time = 0.0 
            
            v = Vehicle(
                v_id=numeric_id,
                pos=(pos[0], pos[1]),
                speed=speed,
                dwell_time_T_stay=dwell_time
            )
            vehicles.append(v)
            
        return vehicles

    def close(self):
        """Safely terminates the TraCI connection."""
        traci.close()
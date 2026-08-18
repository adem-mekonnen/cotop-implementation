import sys
import traci
from typing import Dict

class SumoManager:
    """
    Bridge between the VEC environment and the SUMO traffic simulator.
    Uses TraCI to control the simulation and extract vehicle mobility data.

    Supports a `port`/label so multiple SumoManager instances (e.g. one per
    A3C worker thread) can each own an independent TraCI connection instead
    of colliding on the single global 'default' connection.
    """
    def __init__(self, sumocfg_path: str, port: int = None, use_gui: bool = False):
        self.sumocfg_path = sumocfg_path
        self.use_gui = use_gui
        self.sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        self.port = port
        # Unique label per instance so traci calls don't hit the shared
        # 'default' connection when multiple SumoManagers run concurrently.
        self.label = f"sim_{port}" if port is not None else "default"
        self.conn = None

    def start_simulation(self):
        """Starts the SUMO simulation via TraCI on this instance's own labeled connection."""
        sumo_cmd = [self.sumo_binary, "-c", self.sumocfg_path]
        try:
            traci.start(sumo_cmd, label=self.label)
            self.conn = traci.getConnection(self.label)
            print(f"SUMO simulation started with {self.sumo_binary} (label={self.label})")
        except Exception as e:
            print(f"Failed to start SUMO. Ensure it is installed and in your PATH. Error: {e}")
            sys.exit(1)

    def reload_simulation(self):
        """Reloads the same sumocfg on this instance's existing connection (resets the episode)."""
        if self.conn is None:
            self.start_simulation()
            return
        self.conn.load(["-c", self.sumocfg_path])

    def step(self):
        """Advances the SUMO simulation by one time step."""
        self.conn.simulationStep()

    def get_vehicle_data(self) -> Dict[str, dict]:
        """
        Retrieves active vehicles from SUMO and formats them as a dict keyed
        by vehicle id, matching what VECEnv.reset() expects:
            {v_id: {'pos': (x, y), 'speed': speed}}
        """
        vehicles = {}
        vehicle_ids = self.conn.vehicle.getIDList()

        for v_id_str in vehicle_ids:
            pos = self.conn.vehicle.getPosition(v_id_str)
            speed = self.conn.vehicle.getSpeed(v_id_str)
            vehicles[v_id_str] = {
                "pos": (pos[0], pos[1]),
                "speed": speed,
            }

        return vehicles

    def close(self):
        """Safely terminates this instance's TraCI connection."""
        if self.conn is not None:
            self.conn.close()
        else:
            try:
                traci.close()
            except Exception:
                pass
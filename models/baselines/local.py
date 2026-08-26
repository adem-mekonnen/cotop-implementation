import numpy as np

class LocalPolicy:
    def __init__(self, config=None):
        """
        Local execution strategy (Section V-B).
        Always targets the nearest RSU and forces Case 1 (Standalone).
        """
        self.config = config

    def select_action(self, state: np.ndarray, env_info=None) -> int:
        """
        In our action space, action 0 forces Case 1 (Standalone) at the primary (nearest) RSU.
        """
        return 0

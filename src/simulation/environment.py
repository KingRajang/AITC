"""
Traffic Simulation Environment - FIXED VERSION

CHANGES FROM ORIGINAL:
1. ✅ FIX #1 (Line 3, 37, 117-132): Use RewardCalculator instead of hardcoded scaling
2. ✅ FIX #2 (Lines 15-30, 56-64): Explicit lane mapping instead of arbitrary indexing
3. ✅ FIX #3 (Lines 97-100): NO density simulation - only from vision processing
"""

import numpy as np
from .traffic_models import TrafficModel  # CHANGED: Removed relative import for standalone use
from .reward_calculator import RewardCalculator  # NEW: Import reward calculator

class JammingMachine:
    """
    The main simulation environment for traffic signal control.
    """
    def __init__(self, sim_config: dict, initial_state: dict = None):
        self.config = sim_config
        self.traffic_model = TrafficModel(sim_config)
        self.initial_state_data = initial_state
        
        # ═══════════════════════════════════════════════════════════════
        # FIX #2: EXPLICIT LANE MAPPING
        # OLD CODE (Lines 36-41 in original):
        #   for i, lane in enumerate(self.lanes):
        #       if i < len(lane_ids):
        #           lane_id = lane_ids[i]  # ARBITRARY!
        # 
        # NEW CODE: Load explicit mapping from config
        # ═══════════════════════════════════════════════════════════════
        import json
        import os
        lane_config_path = os.path.join(os.path.dirname(__file__), 'lane_mapping.json')
        if os.path.exists(lane_config_path):
            with open(lane_config_path, 'r') as f:
                lane_config = json.load(f)
                self.lane_mapping = lane_config['lane_mapping']
        else:
            # Fallback if config doesn't exist
            self.lane_mapping = {
                'lane_north': 'North',
                'lane_south': 'South',
                'lane_east': 'East',
                'lane_west': 'West'
            }
        
        self.lanes = ["North", "South", "East", "West"]
        self.num_lanes = len(self.lanes)
        self.max_steps = sim_config['max_steps_per_episode']
        
        # ═══════════════════════════════════════════════════════════════
        # FIX #1: WEIGHTED REWARD CALCULATOR
        # OLD CODE: Hardcoded scaling in _calculate_reward()
        # NEW CODE: Flexible weighted system
        # ═══════════════════════════════════════════════════════════════
        self.reward_calculator = RewardCalculator(sim_config)
        
        self.action_to_signal = {
            0: ("NS", "Short"), 1: ("NS", "Normal"), 2: ("NS", "Long"),
            3: ("EW", "Short"), 4: ("EW", "Normal"), 5: ("EW", "Long")
        }
        
        # DEBUG FLAG
        self.debug_step = 0
        
        self.reset()
        print("Jamming Machine environment initialized.")
        
    def reset(self):
        self.current_step = 0
        self.vehicle_counts = {lane: 0 for lane in self.lanes}
        self.density_scores = {lane: 0.0 for lane in self.lanes}
        self.waiting_times = {lane: 0 for lane in self.lanes}

        # ═══════════════════════════════════════════════════════════════
        # FIX #2 APPLIED: Use explicit lane mapping
        # ═══════════════════════════════════════════════════════════════
        if self.initial_state_data:
            for lane_id, lane_name in self.lane_mapping.items():
                if lane_id in self.initial_state_data:
                    data = self.initial_state_data[lane_id]
                    self.vehicle_counts[lane_name] = data['vehicle_count']
                    self.density_scores[lane_name] = data['density_score']
                else:
                    print(f"Warning: {lane_id} not found in initial_state_data")

        return self._get_state_vector()

    def _get_state_vector(self) -> np.ndarray:
        state = []
        for lane in self.lanes:
            norm_count = min(self.vehicle_counts[lane] / self.config['lane_capacity'], 1.0)
            state.append(norm_count)
            state.append(self.density_scores[lane])
        return np.array(state)

    def step(self, action: int):
        self.current_step += 1
        self.debug_step += 1
        
        signal_phase, _ = self.action_to_signal[action]
        vehicles_departed_total = 0
        
        for lane in self.lanes:
            signal_state = 'GREEN' if (('NS' in signal_phase and lane in ['North', 'South']) or \
                                       ('EW' in signal_phase and lane in ['East', 'West'])) else 'RED'
            
            departures = self.traffic_model.get_vehicle_departures(self.vehicle_counts[lane], signal_state)
            self.vehicle_counts[lane] -= departures
            vehicles_departed_total += departures
            
            arrivals = self.traffic_model.get_vehicle_arrivals()
            self.vehicle_counts[lane] = min(self.vehicle_counts[lane] + arrivals, self.config['lane_capacity'])
            
            # Waiting time tracking (snapshot approach)
            self.waiting_times[lane] = self.vehicle_counts[lane]
            
            # ═══════════════════════════════════════════════════════════════
            # FIX #3: NO DENSITY SIMULATION
            # OLD CODE (Line 74 in original):
            #   self.density_scores[lane] = self.traffic_model.simulate_density_score(...)
            #
            # WHY REMOVED:
            # - Density comes from DBSCAN analysis of real images
            # - Simulating density creates fake/inconsistent data
            # - Training on fake data → agent learns wrong patterns
            # 
            # NEW APPROACH:
            # - Density ONLY set during reset() from initial_state
            # - Represents spatial patterns from computer vision
            # - Does NOT change during simulation steps
            # ═══════════════════════════════════════════════════════════════

        reward = self._calculate_reward(vehicles_departed_total)
        
        # DEBUG: Print first 5 steps
        if self.debug_step <= 5:
            print(f"\n[DEBUG Step {self.debug_step}]")
            print(f"  Departed: {vehicles_departed_total}")
            print(f"  Queues: {self.vehicle_counts}")
            print(f"  Waiting times: {self.waiting_times}")
            print(f"  Reward: {reward:.2f}")
        
        done = self.current_step >= self.max_steps
        next_state = self._get_state_vector()
        
        return next_state, reward, done, {'total_departed': vehicles_departed_total}

    def _calculate_reward(self, vehicles_departed_total: int) -> float:
        """
        ═══════════════════════════════════════════════════════════════
        FIX #1 APPLIED: Use RewardCalculator
        ═══════════════════════════════════════════════════════════════
        OLD CODE (Lines 91-126 in original):
            r_throughput = vehicles_departed_total * 10  # Hardcoded!
            r_queue = -total_queue * 2                   # Hardcoded!
            r_waiting = -avg_queue_per_lane * 3          # Hardcoded!
            r_fairness = -queue_std * 2                  # Hardcoded!
            ...
            return total_reward
        
        NEW CODE:
            Uses RewardCalculator with:
            - Normalized components (0-100 scale)
            - Configurable weights from sim_config.json
            - Easy to adjust for different experiments
        ═══════════════════════════════════════════════════════════════
        """
        metrics = {
            'departed': vehicles_departed_total,
            'vehicle_counts': self.vehicle_counts,
            'waiting_times': self.waiting_times
        }
        
        # Use debug mode for first 5 steps
        debug = (self.debug_step <= 5)
        reward = self.reward_calculator.calculate(metrics, debug=debug)
        
        return reward
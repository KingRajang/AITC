import numpy as np
from collections import deque
from .traffic_models import TrafficModel

class JammingMachine:
    """
    The main simulation environment, as described in Section 3.4 of the thesis.
    This class manages the state of the intersection, processes agent actions,
    and returns rewards.
    """
    def __init__(self, sim_config: dict, initial_state: dict = None):
        """
        Initializes the Jamming Machine environment.
        """
        self.config = sim_config
        self.traffic_model = TrafficModel(sim_config)
        self.initial_state_data = initial_state
        
        # Define the structure of the intersection, assuming 4 approaches (N, S, E, W)
        # We can map specific lane_ids from config to these approaches if needed
        self.lanes = ["North", "South", "East", "West"]
        self.num_lanes = len(self.lanes)
        self.max_steps = sim_config['max_steps_per_episode']
        
        # Action space mapping from thesis Section 3.5.3
        # We simplify by just tracking which directions are green
        self.action_to_signal = {
            0: ("NS", "Short"), 1: ("NS", "Normal"), 2: ("NS", "Long"),
            3: ("EW", "Short"), 4: ("EW", "Normal"), 5: ("EW", "Long")
        }
        
        # Weights for the reward function from Section 3.5.4
        self.reward_weights = {'waiting': 0.4, 'throughput': 0.3, 'queue': 0.2, 'fairness': 0.1}
        
        self.reset()
        print("Jamming Machine environment initialized.")
        
    def reset(self):
        """
        Resets the environment to its initial state.
        """
        self.current_step = 0
        
        # State variables
        self.vehicle_counts = {lane: 0 for lane in self.lanes}
        self.density_scores = {lane: 0.0 for lane in self.lanes}
        self.waiting_times = {lane: 0 for lane in self.lanes} # Total waiting time per lane

        # If an initial state from Phase 1 is provided, use it
        # This part requires mapping the `lane_id` from JSON to the simulator's lanes
        if self.initial_state_data:
            # Simple mapping for now, assuming 4 lanes in the JSON
            # A more robust solution would use the `direction` field in lane_config.json
            lane_ids = list(self.initial_state_data.keys())
            for i, lane in enumerate(self.lanes):
                if i < len(lane_ids):
                    lane_id = lane_ids[i]
                    self.vehicle_counts[lane] = self.initial_state_data[lane_id]['vehicle_count']
                    self.density_scores[lane] = self.initial_state_data[lane_id]['density_score']

        return self._get_state_vector()

    def _get_state_vector(self) -> np.ndarray:
        """
        Constructs the 8-dimensional state vector for the agent.
        As per Section 3.5.2 (normalized).
        """
        state = []
        for lane in self.lanes:
            # Normalize vehicle count
            norm_count = min(self.vehicle_counts[lane] / self.config['lane_capacity'], 1.0)
            state.append(norm_count)
            # Add density score
            state.append(self.density_scores[lane])
        return np.array(state)

    def step(self, action: int):
        """
        Executes one time step in the environment.
        """
        self.current_step += 1
        
        signal_phase, _ = self.action_to_signal[action]
        
        vehicles_departed_total = 0
        
        # --- Update Traffic State ---
        for lane in self.lanes:
            # 1. Vehicle Departures
            signal_state = 'GREEN' if (('NS' in signal_phase and lane in ['North', 'South']) or \
                                       ('EW' in signal_phase and lane in ['East', 'West'])) else 'RED'
            
            departures = self.traffic_model.get_vehicle_departures(self.vehicle_counts[lane], signal_state)
            self.vehicle_counts[lane] -= departures
            vehicles_departed_total += departures
            
            # 2. Vehicle Arrivals
            arrivals = self.traffic_model.get_vehicle_arrivals()
            self.vehicle_counts[lane] = min(self.vehicle_counts[lane] + arrivals, self.config['lane_capacity'])
            
            # 3. Update Waiting Times (add current queue length to total)
            self.waiting_times[lane] += self.vehicle_counts[lane]
            
            # 4. Update Density Score
            self.density_scores[lane] = self.traffic_model.simulate_density_score(self.vehicle_counts[lane])

        # --- Calculate Reward ---
        reward = self._calculate_reward(vehicles_departed_total)
        
        # --- Check for Done ---
        done = self.current_step >= self.max_steps
        
        # --- Get Next State ---
        next_state = self._get_state_vector()
        
        return next_state, reward, done, {}

    def _calculate_reward(self, vehicles_departed_total: int) -> float:
        """
        Calculates the multi-objective reward as per Section 3.5.4.
        """
        w = self.reward_weights
        
        # R_waiting: Negative of total waiting time (we want to minimize this)
        total_waiting_time = sum(self.waiting_times.values())
        total_vehicles = sum(self.vehicle_counts.values()) + 1 # Avoid division by zero
        r_waiting = - (total_waiting_time / total_vehicles)
        
        # R_throughput: Number of vehicles served in this step
        r_throughput = vehicles_departed_total
        
        # R_queue: Negative of sum of queue lengths
        r_queue = - sum(self.vehicle_counts.values())
        
        # R_fairness: Negative standard deviation of per-lane waiting times
        # To make it comparable, we can use normalized waiting times
        avg_wait_per_vehicle = [self.waiting_times[l] / (self.vehicle_counts[l] + 1) for l in self.lanes]
        r_fairness = - np.std(avg_wait_per_vehicle)

        # Combine with weights
        # Note: reward components may need scaling to be on a similar magnitude
        total_reward = (w['waiting'] * r_waiting) + \
                       (w['throughput'] * r_throughput) + \
                       (w['queue'] * r_queue) + \
                       (w['fairness'] * r_fairness)

        return total_reward

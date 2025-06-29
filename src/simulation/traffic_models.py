import numpy as np

class TrafficModel:
    """
    Contains the stochastic models for traffic flow as described in
    Section 3.4.3 of the thesis paper.
    """
    def __init__(self, sim_config: dict):
        """
        Initializes the traffic models with parameters from the simulation config.
        """
        self.arrival_rates = sim_config['traffic_arrival_rates']
        self.lane_capacity = sim_config['lane_capacity']
        self.saturation_flow_rate = sim_config['saturation_flow_rate']
        print("TrafficModel initialized.")

    def get_vehicle_arrivals(self, traffic_level: str = "medium") -> int:
        """
        Calculates vehicle arrivals based on a uniform distribution process.
        As per Section 3.4.3.1.

        Args:
            traffic_level (str): The current traffic scenario ('light', 'medium', 'heavy').

        Returns:
            int: The number of new vehicles arriving in this step.
        """
        min_arrivals, max_arrivals = self.arrival_rates.get(traffic_level, self.arrival_rates['medium'])
        return np.random.randint(min_arrivals, max_arrivals + 1)

    def get_vehicle_departures(self, queue_length: int, signal_state: str) -> int:
        """
        Calculates vehicle departures based on signal state and queue length.
        As per Section 3.4.3.2.

        Args:
            queue_length (int): The number of vehicles currently in the lane.
            signal_state (str): The state of the traffic light ('GREEN', 'YELLOW', 'RED').

        Returns:
            int: The number of vehicles departing in this step.
        """
        if signal_state == 'GREEN':
            # Capacity is based on saturation flow rate. Assuming 1 step = 1 second for simplicity.
            # This can be adjusted. A more complex model could use the durations from the paper.
            departure_capacity = int(self.saturation_flow_rate)
            return min(queue_length, departure_capacity)
        
        # No departures on RED or YELLOW
        return 0

    def simulate_density_score(self, vehicle_count: int) -> float:
        """
        Implements the "Data Faker Logic" to simulate a DBSCAN-like density score.
        As per Section 3.4.3.3.

        Args:
            vehicle_count (int): The current number of vehicles in the lane.

        Returns:
            float: A simulated density score between 0.0 and 1.0.
        """
        fill_ratio = vehicle_count / self.lane_capacity

        # Base classification from the paper
        if fill_ratio <= 0.2:
            base_density = 0.25 # Low
        elif fill_ratio <= 0.5:
            base_density = 0.50 # Medium
        elif fill_ratio <= 0.8:
            base_density = 0.75 # High
        else:
            base_density = 1.0 # Jammed

        # Stochastic Transition: Add a small random noise to mimic spatial variations
        noise = np.random.uniform(-0.05, 0.05)
        simulated_score = np.clip(base_density + noise, 0.0, 1.0)
        
        return round(simulated_score, 2)

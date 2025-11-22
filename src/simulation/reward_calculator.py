"""
Reward Calculator for Traffic Signal Control
Implements weighted, normalized reward components for flexible experimentation.

CHANGELOG:
- NEW FILE: Replaces hardcoded reward scaling in environment.py
- Implements normalized components (0-100 scale)
- Configurable weights from sim_config.json
- Easy to adjust priorities without code changes
"""

import numpy as np

class RewardCalculator:
    """
    Flexible reward calculation with normalized components and configurable weights.
    
    WHY THIS IS BETTER THAN HARDCODED SCALING:
    1. Research flexibility: Change priorities via config, no code changes
    2. Fair comparison: All components normalized to same scale (0-100)
    3. Interpretability: Can analyze each component's contribution
    4. Reproducibility: Document exact weights used in experiments
    """
    
    def __init__(self, config: dict):
        """
        Initialize reward calculator with configuration.
        
        Args:
            config: Dictionary containing:
                - lane_capacity: Maximum vehicles per lane
                - reward_weights: Dict with component weights
                - normalization_params: Optional custom normalization values
        """
        # Normalization parameters (expected maximum values)
        norm_params = config.get('normalization_params', {})
        self.max_throughput = norm_params.get('max_throughput', 20)
        self.max_queue = config.get('lane_capacity', 50)
        self.max_waiting = norm_params.get('max_waiting', 100)
        self.max_queue_std = norm_params.get('max_queue_std', 20)
        
        # Component weights (must sum to 1.0 for interpretability)
        weights = config.get('reward_weights', {})
        self.w_throughput = weights.get('throughput', 0.4)
        self.w_queue = weights.get('queue', 0.3)
        self.w_waiting = weights.get('waiting', 0.2)
        self.w_fairness = weights.get('fairness', 0.1)
        
        # Validate weights sum to 1.0
        total_weight = self.w_throughput + self.w_queue + self.w_waiting + self.w_fairness
        if abs(total_weight - 1.0) > 0.01:
            print(f"Warning: Weights sum to {total_weight:.3f}, not 1.0. Normalizing...")
            self.w_throughput /= total_weight
            self.w_queue /= total_weight
            self.w_waiting /= total_weight
            self.w_fairness /= total_weight
        
        print(f"RewardCalculator initialized with weights:")
        print(f"  Throughput: {self.w_throughput:.2f}")
        print(f"  Queue:      {self.w_queue:.2f}")
        print(f"  Waiting:    {self.w_waiting:.2f}")
        print(f"  Fairness:   {self.w_fairness:.2f}")
    
    def calculate(self, metrics: dict, debug: bool = False) -> float:
        """
        Calculate total reward from traffic metrics.
        
        REWARD COMPONENTS:
        1. Throughput (positive): More vehicles departing = better
        2. Queue (negative): Fewer vehicles waiting = better
        3. Waiting time (negative): Lower average wait = better
        4. Fairness (negative): More balanced queues = better
        
        Args:
            metrics: Dictionary containing:
                - departed: Number of vehicles that departed
                - vehicle_counts: Dict of current queue per lane
                - waiting_times: Dict of waiting times per lane
            debug: If True, print component breakdown
        
        Returns:
            Total weighted reward (scale: 0-100)
        """
        # Extract metrics
        departed = metrics.get('departed', 0)
        vehicle_counts = metrics.get('vehicle_counts', {})
        waiting_times = metrics.get('waiting_times', {})
        
        total_queue = sum(vehicle_counts.values())
        num_lanes = len(vehicle_counts)
        
        # Component 1: Throughput (POSITIVE - higher is better)
        # Normalized to 0-100 scale
        r_throughput = min((departed / self.max_throughput) * 100, 100)
        
        # Component 2: Queue penalty (NEGATIVE - convert to positive scale)
        # Inverted: low queue = high reward
        r_queue = max((1 - total_queue / (self.max_queue * num_lanes)) * 100, 0)
        
        # Component 3: Waiting time penalty (NEGATIVE - convert to positive scale)
        # Average waiting time across all lanes
        if isinstance(waiting_times, dict):
            avg_waiting = sum(waiting_times.values()) / max(1, num_lanes)
        else:
            avg_waiting = waiting_times
        r_waiting = max((1 - avg_waiting / self.max_waiting) * 100, 0)
        
        # Component 4: Fairness (NEGATIVE - convert to positive scale)
        # Low standard deviation = high fairness
        if num_lanes > 1:
            queue_values = list(vehicle_counts.values())
            queue_std = np.std(queue_values)
            r_fairness = max((1 - queue_std / self.max_queue_std) * 100, 0)
        else:
            r_fairness = 100  # Perfect fairness for single lane
        
        # Apply weights and calculate total
        total_reward = (
            self.w_throughput * r_throughput +
            self.w_queue * r_queue +
            self.w_waiting * r_waiting +
            self.w_fairness * r_fairness
        )
        
        # Debug output
        if debug:
            print(f"\n[Reward Breakdown]")
            print(f"  Throughput: {r_throughput:6.2f} (weight: {self.w_throughput:.2f}) = {self.w_throughput * r_throughput:6.2f}")
            print(f"  Queue:      {r_queue:6.2f} (weight: {self.w_queue:.2f}) = {self.w_queue * r_queue:6.2f}")
            print(f"  Waiting:    {r_waiting:6.2f} (weight: {self.w_waiting:.2f}) = {self.w_waiting * r_waiting:6.2f}")
            print(f"  Fairness:   {r_fairness:6.2f} (weight: {self.w_fairness:.2f}) = {self.w_fairness * r_fairness:6.2f}")
            print(f"  TOTAL:      {total_reward:6.2f}")
        
        return total_reward
    
    def get_component_scores(self, metrics: dict) -> dict:
        """
        Get individual component scores for analysis.
        
        Returns:
            Dictionary with raw component scores (0-100 scale)
        """
        departed = metrics.get('departed', 0)
        vehicle_counts = metrics.get('vehicle_counts', {})
        waiting_times = metrics.get('waiting_times', {})
        
        total_queue = sum(vehicle_counts.values())
        num_lanes = len(vehicle_counts)
        
        r_throughput = min((departed / self.max_throughput) * 100, 100)
        r_queue = max((1 - total_queue / (self.max_queue * num_lanes)) * 100, 0)
        
        if isinstance(waiting_times, dict):
            avg_waiting = sum(waiting_times.values()) / max(1, num_lanes)
        else:
            avg_waiting = waiting_times
        r_waiting = max((1 - avg_waiting / self.max_waiting) * 100, 0)
        
        if num_lanes > 1:
            queue_values = list(vehicle_counts.values())
            queue_std = np.std(queue_values)
            r_fairness = max((1 - queue_std / self.max_queue_std) * 100, 0)
        else:
            r_fairness = 100
        
        return {
            'throughput': r_throughput,
            'queue': r_queue,
            'waiting': r_waiting,
            'fairness': r_fairness,
            'weighted_throughput': self.w_throughput * r_throughput,
            'weighted_queue': self.w_queue * r_queue,
            'weighted_waiting': self.w_waiting * r_waiting,
            'weighted_fairness': self.w_fairness * r_fairness
        }
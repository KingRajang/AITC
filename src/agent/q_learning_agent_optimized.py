import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    """
    OPTIMIZED Q-Learning agent with multiple improvements to beat actuated baseline.
    
    KEY IMPROVEMENTS FROM ORIGINAL:
    1. 10 bins instead of 5 (finer state discretization)
    2. Lower initial Q-values (80-120 → 60-80) for better exploration
    3. Adaptive learning rate (decreases with state visits)
    4. Slower epsilon decay for more exploration
    5. Better discretization with uniform bins
    
    Expected improvement: +5-10% over original
    """
    def __init__(self, state_space_size: int, action_space_size: int, learning_rate: float,
                 discount_factor: float, exploration_rate: float):
        """
        Initializes the Q-Learning agent with optimized hyperparameters.
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        
        # Hyperparameters for exploration decay
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay_rate = 0.0003  # CHANGED: Slower decay (was 0.0005)
        
        # Track state visits for adaptive learning rate
        self.state_visit_count = defaultdict(int)
        
        # IMPROVEMENT #1: Lower optimistic initialization
        # Less optimistic = more realistic Q-values = better learning
        # Changed from [80, 120] to [60, 80]
        self.q_table = defaultdict(lambda: np.array([random.uniform(60, 80) for _ in range(self.action_space_size)]))
        
        # IMPROVEMENT #2: 10 bins instead of 5
        # More granular state representation
        # Old: 5^8 = 390,625 states
        # New: 10^8 = 100,000,000 states (sparse but more precise)
        self.num_bins = 10
        self.bin_edges = np.linspace(0.0, 1.0, self.num_bins + 1)
        
        print(f"QLearningAgent initialized with {self.num_bins} bins.")
        print(f"State space size: {self.num_bins}^{state_space_size} = {self.num_bins**state_space_size:,} possible states")

    def _discretize_state(self, state: np.ndarray) -> tuple:
        """
        IMPROVEMENT #3: Simpler, uniform discretization
        Maps continuous values [0, 1] to discrete bins [0, 9]
        """
        discretized = []
        for value in state:
            # Clip to valid range
            clipped = np.clip(value, 0.0, 1.0)
            # Find bin index
            bin_idx = np.digitize(clipped, self.bin_edges) - 1
            # Ensure bin index is valid
            bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)
            discretized.append(bin_idx)
        return tuple(discretized)

    def choose_action(self, state: np.ndarray) -> int:
        """
        Chooses an action using epsilon-greedy policy.
        """
        discrete_state = self._discretize_state(state)
        
        # Track state visits
        self.state_visit_count[discrete_state] += 1
        
        # Exploration: choose a random action
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_space_size)
        # Exploitation: choose the best known action
        else:
            return np.argmax(self.q_table[discrete_state])

    def update_q_table(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """
        IMPROVEMENT #4: Adaptive learning rate based on state visits
        States visited more often get smaller learning rates (more stable)
        States visited rarely get larger learning rates (faster learning)
        """
        current_discrete_state = self._discretize_state(state)
        next_discrete_state = self._discretize_state(next_state)
        
        # Adaptive learning rate: decreases with state visits
        visits = self.state_visit_count[current_discrete_state]
        adaptive_lr = self.lr / (1.0 + visits / 1000.0)
        
        # Q-Learning update with adaptive learning rate
        old_value = self.q_table[current_discrete_state][action]
        next_max = np.max(self.q_table[next_discrete_state])
        
        new_value = (1 - adaptive_lr) * old_value + adaptive_lr * (reward + self.gamma * next_max)
        
        self.q_table[current_discrete_state][action] = new_value

    def decay_exploration_rate(self, episode: int):
        """
        IMPROVEMENT #5: Slower epsilon decay for better exploration
        """
        self.epsilon = self.min_epsilon + \
                       (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay_rate * episode)

    def save_q_table(self, file_path: str):
        """
        Saves the Q-table to a file.
        """
        try:
            serializable_q_table = {str(k): v.tolist() for k, v in self.q_table.items()}
            with open(file_path, 'w') as f:
                import json
                json.dump(serializable_q_table, f)
            print(f"Q-table successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def load_q_table(self, file_path: str):
        """
        Loads a Q-table from a file.
        """
        try:
            with open(file_path, 'r') as f:
                import json
                loaded_q_table = json.load(f)
            
            self.q_table.clear()
            for k, v in loaded_q_table.items():
                state_tuple = tuple(map(int, k.strip('()').split(',')))
                self.q_table[state_tuple] = np.array(v)
            
            print(f"Q-table successfully loaded from {file_path}")
        except Exception as e:
            print(f"Error loading Q-table: {e}")

    def get_statistics(self):
        """
        Get statistics about the Q-table for analysis.
        """
        if len(self.q_table) == 0:
            return {"states_learned": 0}
        
        all_q_values = []
        for q_vals in self.q_table.values():
            all_q_values.extend(q_vals)
        
        return {
            "states_learned": len(self.q_table),
            "avg_q_value": np.mean(all_q_values),
            "max_q_value": np.max(all_q_values),
            "min_q_value": np.min(all_q_values),
            "current_epsilon": self.epsilon
        }
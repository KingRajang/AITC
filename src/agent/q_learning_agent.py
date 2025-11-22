import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    """
    HYBRID Q-Learning agent: Keep 5 bins but add other improvements
    
    This combines:
    - 5 bins (like original) - manageable state space
    - Adaptive learning rate - NEW
    - Slower epsilon decay - NEW  
    - Lower initial Q-values - NEW
    - Fixed save/load bugs - NEW
    """
    def __init__(self, state_space_size: int, action_space_size: int, learning_rate: float,
                 discount_factor: float, exploration_rate: float):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay_rate = 0.0004  # Slightly slower than original 0.0005
        
        # Track state visits for adaptive learning rate
        self.state_visit_count = defaultdict(int)
        
        # Lower initial Q-values (60-80 instead of 80-120)
        self.q_table = defaultdict(lambda: np.array([random.uniform(60, 80) for _ in range(self.action_space_size)]))
        
        # KEEP 5 BINS (proven to work better than 10)
        self.bin_edges = [0.0, 0.10, 0.20, 0.30, 0.50, 1.0]
        self.num_bins = len(self.bin_edges) - 1
        
        print(f"QLearningAgent initialized with {self.num_bins} bins.")
        print(f"State space size: {self.num_bins}^{state_space_size} = {self.num_bins**state_space_size:,} possible states")

    def _discretize_state(self, state: np.ndarray) -> tuple:
        """Use original bin edges that worked well"""
        discretized = []
        for value in state:
            bin_index = 0
            for i in range(len(self.bin_edges) - 1):
                if self.bin_edges[i] <= value < self.bin_edges[i + 1]:
                    bin_index = i
                    break
            if value >= self.bin_edges[-1]:
                bin_index = len(self.bin_edges) - 2
            discretized.append(bin_index)
        return tuple(discretized)

    def choose_action(self, state: np.ndarray) -> int:
        discrete_state = self._discretize_state(state)
        
        # Track visits
        self.state_visit_count[discrete_state] += 1
        
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_space_size)
        else:
            return np.argmax(self.q_table[discrete_state])

    def update_q_table(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """
        NEW: Adaptive learning rate based on visits
        """
        current_discrete_state = self._discretize_state(state)
        next_discrete_state = self._discretize_state(next_state)
        
        # Adaptive learning rate
        visits = self.state_visit_count[current_discrete_state]
        adaptive_lr = self.lr / (1.0 + visits / 1000.0)
        
        old_value = self.q_table[current_discrete_state][action]
        next_max = np.max(self.q_table[next_discrete_state])
        
        new_value = (1 - adaptive_lr) * old_value + adaptive_lr * (reward + self.gamma * next_max)
        
        self.q_table[current_discrete_state][action] = new_value

    def decay_exploration_rate(self, episode: int):
        """Slightly slower decay"""
        self.epsilon = self.min_epsilon + \
                       (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay_rate * episode)

    def save_q_table(self, file_path: str):
        """FIXED: Properly handle numpy types"""
        try:
            serializable_q_table = {}
            for k, v in self.q_table.items():
                # Convert numpy int64 to regular Python int
                key_str = str(tuple(int(x) for x in k))
                serializable_q_table[key_str] = v.tolist()
            
            with open(file_path, 'w') as f:
                import json
                json.dump(serializable_q_table, f)
            print(f"Q-table successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def load_q_table(self, file_path: str):
        """FIXED: Properly parse keys"""
        try:
            with open(file_path, 'r') as f:
                import json
                loaded_q_table = json.load(f)
            
            self.q_table.clear()
            for k, v in loaded_q_table.items():
                k_clean = k.strip('()').replace(' ', '')
                if k_clean:
                    state_tuple = tuple(int(x) for x in k_clean.split(','))
                    self.q_table[state_tuple] = np.array(v)
            
            print(f"Q-table successfully loaded from {file_path}")
            print(f"Loaded {len(self.q_table):,} states")
        except Exception as e:
            print(f"Error loading Q-table: {e}")
            import traceback
            traceback.print_exc()

    def get_statistics(self):
        """Get stats"""
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
import numpy as np
from collections import defaultdict

class QLearningAgent:
    """
    Implements the Q-Learning agent as described in Section 3.5 of the thesis.
    This agent learns a policy to control traffic signals by interacting with
    the "Jamming Machine" environment.
    """
    def __init__(self, state_space_size: int, action_space_size: int, learning_rate: float,
                 discount_factor: float, exploration_rate: float):
        """
        Initializes the Q-Learning agent with hyperparameters from Section 3.5.5.
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        
        # Hyperparameters for exploration decay
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay_rate = 0.0005 # A parameter to tune based on training speed

        # The Q-table. We use a defaultdict for convenience, so we don't have to
        # check if a state has been seen before. It will default to a zero array.
        # However, states from the environment are continuous, so we must discretize them
        # to use them as dictionary keys.
        self.q_table = defaultdict(lambda: np.zeros(self.action_space_size))
        
        # Define the number of bins for discretizing the continuous state space
        self.state_bins = [10] * self.state_space_size # 10 bins for each of the 8 dimensions

        print("QLearningAgent initialized.")

    def _discretize_state(self, state: np.ndarray) -> tuple:
        """
        Converts a continuous state vector into a discrete tuple for Q-table keys.
        The state values are normalized between 0.0 and 1.0 by the environment.
        """
        discretized = []
        for i, value in enumerate(state):
            # Scale the value to the number of bins and convert to an integer
            bin_index = int(value * (self.state_bins[i] - 1))
            discretized.append(bin_index)
        return tuple(discretized)

    def choose_action(self, state: np.ndarray) -> int:
        """
        Chooses an action using an epsilon-greedy policy.
        """
        discrete_state = self._discretize_state(state)
        
        # Exploration: choose a random action
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_space_size)
        # Exploitation: choose the best known action for the current state
        else:
            return np.argmax(self.q_table[discrete_state])

    def update_q_table(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """
        Updates the Q-table using the Bellman equation (Q-Learning update rule).
        As per Section 3.5.5.
        """
        current_discrete_state = self._discretize_state(state)
        next_discrete_state = self._discretize_state(next_state)
        
        # The Q-Learning formula
        old_value = self.q_table[current_discrete_state][action]
        next_max = np.max(self.q_table[next_discrete_state])
        
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        
        self.q_table[current_discrete_state][action] = new_value

    def decay_exploration_rate(self, episode: int):
        """
        Decays the epsilon value exponentially to shift from exploration to exploitation.
        As per Section 3.5.5.
        """
        self.epsilon = self.min_epsilon + \
                       (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay_rate * episode)

    def save_q_table(self, file_path: str):
        """
        Saves the Q-table to a file.
        Since defaultdict is not directly serializable with numpy, we convert it to a regular dict first.
        """
        try:
            # Convert defaultdict to a regular dict for saving
            serializable_q_table = {str(k): v.tolist() for k, v in self.q_table.items()}
            with open(file_path, 'w') as f:
                import json
                json.dump(serializable_q_table, f)
            print(f"Q-table successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def load_q_table(self, file_path: str):
        """
        Loads a Q-table from a file for evaluation or continued training.
        """
        try:
            with open(file_path, 'r') as f:
                import json
                loaded_q_table = json.load(f)
            
            # Convert keys back to tuples and values back to numpy arrays
            self.q_table.clear()
            for k, v in loaded_q_table.items():
                # The key was saved as a string of a tuple, e.g., "(0, 1, 2, 3...)"
                # We need to parse it back into a tuple of integers.
                state_tuple = tuple(map(int, k.strip('()').split(',')))
                self.q_table[state_tuple] = np.array(v)
            
            print(f"Q-table successfully loaded from {file_path}")
        except Exception as e:
            print(f"Error loading Q-table: {e}")

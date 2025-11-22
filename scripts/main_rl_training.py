# main_rl_training.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import numpy as np
from src.simulation.environment import JammingMachine
from src.agent.q_learning_agent import QLearningAgent

# --- Configuration ---
SIM_CONFIG_PATH = "config/sim_config.json"
INITIAL_STATE_PATH = "data/initial_state.json"
LOG_FILE_PATH = "training_log.json" # <-- NEW: Path for the log file

def main():
    """
    Entrypoint for Phase 2 & 3: Simulation and RL Training.
    This script trains the Q-Learning agent in the "Jamming Machine" environment.
    """
    print("--- Phase 2 & 3: RL Agent Training (with Logging) ---")

    # --- Load Configuration ---
    with open(SIM_CONFIG_PATH) as f:
        sim_config = json.load(f)
    
    initial_state = None
    try:
        with open(INITIAL_STATE_PATH) as f:
            initial_state = json.load(f)
        print("Loaded initial state from Phase 1 analysis.")
    except FileNotFoundError:
        print("No initial state file found. Simulator will start from a default state.")

    # Scale down initial state to manageable levels
    if initial_state:
        print("Scaling down initial state...")
        for lane_id in initial_state:
            original = initial_state[lane_id]['vehicle_count']
            initial_state[lane_id]['vehicle_count'] = initial_state[lane_id]['vehicle_count'] // 3
            print(f"  {lane_id}: {original} → {initial_state[lane_id]['vehicle_count']} cars")

    # --- Initialization ---
    env = JammingMachine(sim_config, initial_state)
    agent = QLearningAgent(
        state_space_size=8, 
        action_space_size=6,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0
    )
    print("Jamming Machine environment and Q-Learning agent initialized.")

    # --- Training Loop ---
    print(f"Starting training for {sim_config['simulation_episodes']} episodes...")
    training_log = [] # <-- NEW: List to store log data
    for episode in range(sim_config['simulation_episodes']):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            step_count += 1

        agent.decay_exploration_rate(episode)
        
        # <-- NEW: Log data for this episode -->
        log_entry = {
            'episode': episode + 1, 
            'total_reward': total_reward,
            'avg_reward_per_step': total_reward / step_count,  # <<< ADD THIS
            'epsilon': agent.epsilon
        }
        training_log.append(log_entry)

        if (episode + 1) % 100 == 0:
            avg_queue = sum(env.vehicle_counts.values()) / 4  # <<< ADD THIS
            print(f"Episode {episode + 1}/{sim_config['simulation_episodes']} | "
                  f"Total Reward: {total_reward:.2f} | "
                  f"Avg/Step: {total_reward/step_count:.2f} | "  # <<< ADD THIS
                  f"Avg Queue: {avg_queue:.1f} | "  # <<< ADD THIS
                  f"Epsilon: {agent.epsilon:.4f}")

    # --- Save Trained Model and Log ---
    agent.save_q_table("trained_q_table.json") # Corrected extension
    
    # <-- NEW: Save the log file -->
    with open(LOG_FILE_PATH, 'w') as f:
        json.dump(training_log, f, indent=4)
    print(f"\nTraining log saved to {LOG_FILE_PATH}")
    
    print("\nTraining complete.")
    print("\n--- Phase 2 & 3 Complete ---")

if __name__ == "__main__":
    main()

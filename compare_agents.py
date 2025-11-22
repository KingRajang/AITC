#!/usr/bin/env python3
"""
QUICK COMPARISON: Old Agent vs Optimized Agent
This will train both for 2000 episodes and compare results
"""

import sys 
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add project root to path
sys.path.insert(0, SCRIPT_DIR)

from src.simulation.environment import JammingMachine

# Import both agents
from src.agent.q_learning_agent import QLearningAgent as OldAgent

# Copy the optimized agent code here for testing
import random
from collections import defaultdict

class OptimizedAgent:
    """Optimized Q-Learning agent"""
    def __init__(self, state_space_size: int, action_space_size: int, learning_rate: float,
                 discount_factor: float, exploration_rate: float):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay_rate = 0.0003  # Slower decay
        
        self.state_visit_count = defaultdict(int)
        self.q_table = defaultdict(lambda: np.array([random.uniform(60, 80) for _ in range(self.action_space_size)]))
        
        # 10 bins instead of 5
        self.num_bins = 10
        self.bin_edges = np.linspace(0.0, 1.0, self.num_bins + 1)
        print(f"✅ Optimized Agent: {self.num_bins} bins")
        
    def _discretize_state(self, state: np.ndarray) -> tuple:
        discretized = []
        for value in state:
            clipped = np.clip(value, 0.0, 1.0)
            bin_idx = np.digitize(clipped, self.bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)
            discretized.append(bin_idx)
        return tuple(discretized)

    def choose_action(self, state: np.ndarray) -> int:
        discrete_state = self._discretize_state(state)
        self.state_visit_count[discrete_state] += 1
        
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_space_size)
        else:
            return np.argmax(self.q_table[discrete_state])

    def update_q_table(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
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
        self.epsilon = self.min_epsilon + \
                       (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay_rate * episode)


def train_agent(agent, env, num_episodes=2000, agent_name="Agent"):
    """Train an agent and return episode rewards"""
    print(f"\n{'='*70}")
    print(f"Training {agent_name} for {num_episodes} episodes...")
    print(f"{'='*70}")
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        agent.decay_exploration_rate(episode)
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 200 == 0:
            recent_avg = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Recent Avg: {recent_avg:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    return episode_rewards


def evaluate_agent(agent, env, num_episodes=50, agent_name="Agent"):
    """Evaluate trained agent"""
    print(f"\n{'='*70}")
    print(f"Evaluating {agent_name}...")
    print(f"{'='*70}")
    
    # Set epsilon to 0 for pure exploitation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
        
        rewards.append(total_reward)
    
    agent.epsilon = original_epsilon
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    print(f"\n{agent_name} Results:")
    print(f"  Mean: {mean_reward:.2f}")
    print(f"  Std:  {std_reward:.2f}")
    print(f"  Q-table size: {len(agent.q_table):,} states")
    
    return mean_reward, std_reward


def main():
    print("="*70)
    print("COMPARING OLD vs OPTIMIZED Q-LEARNING AGENT")
    print("="*70)
    
    # Load config from correct location
    config_path = os.path.join(SCRIPT_DIR, 'config', 'sim_config.json')
    if not os.path.exists(config_path):
        config_path = os.path.join(SCRIPT_DIR, 'sim_config.json')
    
    print(f"\nLoading config from: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"❌ ERROR: Could not find sim_config.json")
        print(f"   Looked in:")
        print(f"   - {os.path.join(SCRIPT_DIR, 'config', 'sim_config.json')}")
        print(f"   - {os.path.join(SCRIPT_DIR, 'sim_config.json')}")
        print(f"\n   Current directory: {os.getcwd()}")
        print(f"   Script directory: {SCRIPT_DIR}")
        return
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Reduce episodes for quick test
    config['simulation_episodes'] = 2000
    
    print("✅ Config loaded")
    
    # Create environment
    env = JammingMachine(config)
    
    # Train OLD agent
    print("\n🔵 TRAINING OLD AGENT (5 bins, original settings)")
    old_agent = OldAgent(8, 6, 0.1, 0.95, 1.0)
    old_rewards = train_agent(old_agent, env, 2000, "OLD Agent")
    
    # Train OPTIMIZED agent
    print("\n🟢 TRAINING OPTIMIZED AGENT (10 bins, adaptive LR, slower decay)")
    env_opt = JammingMachine(config)  # Fresh environment
    opt_agent = OptimizedAgent(8, 6, 0.1, 0.95, 1.0)
    opt_rewards = train_agent(opt_agent, env_opt, 2000, "OPTIMIZED Agent")
    
    # Evaluate both
    env_eval = JammingMachine(config)
    old_mean, old_std = evaluate_agent(old_agent, env_eval, 50, "OLD Agent")
    
    env_eval2 = JammingMachine(config)
    opt_mean, opt_std = evaluate_agent(opt_agent, env_eval2, 50, "OPTIMIZED Agent")
    
    # Compare
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    improvement = ((opt_mean - old_mean) / old_mean) * 100
    
    print(f"\nOLD Agent:       {old_mean:.2f} ± {old_std:.2f}")
    print(f"OPTIMIZED Agent: {opt_mean:.2f} ± {opt_std:.2f}")
    print(f"\nImprovement: {improvement:+.2f}%")
    print(f"\nACTUATED BASELINE: 21,842 (from your results)")
    print(f"GAP TO CLOSE: 5.4%")
    
    if improvement > 2:
        print("\n✅ OPTIMIZED AGENT IS BETTER!")
        if improvement >= 5:
            print("💪 This improvement should beat actuated!")
        else:
            print("😊 Good progress, might need full training to beat actuated")
    elif improvement > 0:
        print("\n😊 OPTIMIZED AGENT SLIGHTLY BETTER")
        print("💡 Try full 10K episode training")
    else:
        print("\n⚠️  OLD AGENT STILL BETTER (try more training)")
    
    # Plot learning curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    window = 50
    old_smooth = np.convolve(old_rewards, np.ones(window)/window, mode='valid')
    opt_smooth = np.convolve(opt_rewards, np.ones(window)/window, mode='valid')
    
    plt.plot(old_smooth, label='OLD Agent (5 bins)', alpha=0.8)
    plt.plot(opt_smooth, label='OPTIMIZED Agent (10 bins)', alpha=0.8)
    plt.xlabel('Episode')
    plt.ylabel('Reward (smoothed)')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(['OLD\n(5 bins)', 'OPTIMIZED\n(10 bins)'], [old_mean, opt_mean], 
            yerr=[old_std, opt_std], capsize=10,
            color=['#1976D2', '#2E7D32'], alpha=0.7)
    plt.ylabel('Mean Reward')
    plt.title('Final Performance')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(SCRIPT_DIR, 'old_vs_optimized_comparison.png')
    plt.savefig(output_path, dpi=300)
    print(f"\n📊 Plot saved to: {output_path}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    if improvement > 2:
        print("1. ✅ Replace your old agent with the optimized one")
        print("2. 🚀 Train for full 10K episodes")
        print("3. 📊 Compare against actuated baseline")
        print("4. 🎯 Expected: Should beat or match actuated!")
    else:
        print("1. 🔍 Need more training episodes (try 5K)")
        print("2. 🎲 Try different reward weights in sim_config.json")
        print("3. 📈 Consider adding temporal features")


if __name__ == "__main__":
    main()
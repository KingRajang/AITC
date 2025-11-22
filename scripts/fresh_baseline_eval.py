import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import sys
import numpy as np
from scipy import stats


from src.simulation.environment import JammingMachine
from src.agent.q_learning_agent import QLearningAgent

print("="*70)
print("FRESH BASELINE EVALUATION - NEW Q-TABLE")
print("="*70)

# Load config
with open('config/sim_config.json') as f:
    config = json.load(f)

# Create environment
env = JammingMachine(config)

# Load Q-table (FRESH!)
print("\n🔄 Loading Q-table from disk (forced fresh load)...")
agent = QLearningAgent(8, 6, 0, 0.95, 0)
agent.load_q_table('trained_q_table.json')

print(f"✅ Q-table loaded: {len(agent.q_table)} states")

# Check exploration rate (handle different attribute names)
try:
    if hasattr(agent, 'exploration_rate'):
        print(f"   Exploration rate: {agent.exploration_rate} (should be 0.0)")
    elif hasattr(agent, 'epsilon'):
        print(f"   Exploration rate (epsilon): {agent.epsilon} (should be 0.0)")
except:
    print("   Exploration rate: (attribute not found)")

# Sample Q-values to verify
sample_state = list(agent.q_table.keys())[0]
print(f"   Sample Q-values: {[f'{v:.2f}' for v in agent.q_table[sample_state]]}")

# Baseline agent
class FixedTimeAgent:
    def __init__(self):
        self.action_sequence = [1, 4]
        self.steps_per_action = 30
        self.current_step = 0
        self.action_index = 0
    
    def choose_action(self, state):
        if self.current_step > 0 and self.current_step % self.steps_per_action == 0:
            self.action_index = (self.action_index + 1) % len(self.action_sequence)
        self.current_step += 1
        return self.action_sequence[self.action_index]
    
    def reset(self):
        self.current_step = 0
        self.action_index = 0

baseline = FixedTimeAgent()

# Evaluate both agents
NUM_EPISODES = 50  # Quick evaluation

print("\n" + "="*70)
print("EVALUATING Q-LEARNING AGENT")
print("="*70)

rl_rewards = []
for ep in range(NUM_EPISODES):
    state = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = agent.choose_action(state)
        state, reward, done, info = env.step(action)
        episode_reward += reward
    
    rl_rewards.append(episode_reward)
    
    if (ep + 1) % 10 == 0:
        print(f"  Episode {ep+1}/{NUM_EPISODES}: Reward = {episode_reward:.2f}")

rl_mean = np.mean(rl_rewards)
rl_std = np.std(rl_rewards, ddof=1)

print(f"\n✅ Q-Learning Results:")
print(f"   Mean Reward: {rl_mean:.2f} ± {rl_std:.2f}")
print(f"   Range: [{min(rl_rewards):.2f}, {max(rl_rewards):.2f}]")

print("\n" + "="*70)
print("EVALUATING FIXED-TIME BASELINE")
print("="*70)

baseline_rewards = []
for ep in range(NUM_EPISODES):
    state = env.reset()
    baseline.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = baseline.choose_action(state)
        state, reward, done, info = env.step(action)
        episode_reward += reward
    
    baseline_rewards.append(episode_reward)
    
    if (ep + 1) % 10 == 0:
        print(f"  Episode {ep+1}/{NUM_EPISODES}: Reward = {episode_reward:.2f}")

baseline_mean = np.mean(baseline_rewards)
baseline_std = np.std(baseline_rewards, ddof=1)

print(f"\n✅ Baseline Results:")
print(f"   Mean Reward: {baseline_mean:.2f} ± {baseline_std:.2f}")
print(f"   Range: [{min(baseline_rewards):.2f}, {max(baseline_rewards):.2f}]")

# Statistical comparison
print("\n" + "="*70)
print("STATISTICAL COMPARISON")
print("="*70)

improvement = (rl_mean - baseline_mean) / baseline_mean * 100
t_stat, p_value = stats.ttest_ind(rl_rewards, baseline_rewards)

print(f"\nQ-Learning: {rl_mean:.2f} ± {rl_std:.2f}")
print(f"Baseline:   {baseline_mean:.2f} ± {baseline_std:.2f}")
print(f"\nImprovement: {improvement:+.2f}%")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.05:
    if improvement > 0:
        print("\n🎉 Q-Learning is SIGNIFICANTLY BETTER! (p < 0.05)")
    else:
        print("\n⚠️  Baseline is significantly better (p < 0.05)")
else:
    print("\n⚠️  No significant difference (p >= 0.05)")

print("="*70)

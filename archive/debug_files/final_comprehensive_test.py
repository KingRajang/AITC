import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import sys
import numpy as np
import json
from src.simulation.environment import JammingMachine
from src.agent.q_learning_agent import QLearningAgent

print("=" * 70)
print("🏆 FINAL COMPREHENSIVE COMPARISON 🏆")
print("=" * 70)

with open('config/sim_config.json') as f:
    config = json.load(f)

results = {}

# Test all baselines
intervals = [15, 20, 25, 30, 35, 40, 45]
for interval in intervals:
    print(f"\nTesting {interval}s baseline...")
    env = JammingMachine(config)
    rewards = []
    
    for ep in range(20):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            action = 0 if step % (interval * 2) < interval else 3
            state, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1
        
        rewards.append(episode_reward)
    
    results[f'{interval}s baseline'] = {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'per_step': np.mean(rewards) / 300
    }

# RL test
print(f"\nTesting RL agent (NEW)...")
env = JammingMachine(config)
agent = QLearningAgent(8, 6, 0, 0.95, 0)
agent.load_q_table('trained_q_table.json')
agent.epsilon = 0

rl_rewards = []
for ep in range(20):
    state = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = agent.choose_action(state)
        state, reward, done, info = env.step(action)
        episode_reward += reward
    
    rl_rewards.append(episode_reward)

results['RL (NEW)'] = {
    'mean': np.mean(rl_rewards),
    'std': np.std(rl_rewards),
    'per_step': np.mean(rl_rewards) / 300
}

# Display results
print("\n" + "=" * 70)
print("🎯 FINAL RESULTS COMPARISON 🎯")
print("=" * 70)

print(f"\n{'Method':<20} {'Performance':<15} {'vs Best':<15} {'Status':<20}")
print("-" * 70)

best_perf = max(r['per_step'] for r in results.values())
sorted_results = sorted(results.items(), key=lambda x: x[1]['per_step'], reverse=True)

for method, data in sorted_results:
    perf = data['per_step']
    diff = ((perf - best_perf) / best_perf) * 100
    
    if method == 'RL (NEW)':
        marker = " 🏆 RL!" if perf == best_perf else " 🎉 RL"
    elif perf == best_perf:
        marker = " ← BEST"
    else:
        marker = ""
    
    print(f"{method:<20} {perf:.2f}/step{' '*5} {diff:+.1f}%{marker}")

print("\n" + "=" * 70)

# Find RL rank
rl_perf = results['RL (NEW)']['per_step']
rl_rank = [i for i, (m, _) in enumerate(sorted_results) if m == 'RL (NEW)'][0] + 1
best_baseline = max(v['per_step'] for k, v in results.items() if 'baseline' in k)
best_baseline_name = [k for k, v in results.items() if 'baseline' in k and v['per_step'] == best_baseline][0]

print("📊 RL PERFORMANCE SUMMARY:")
print(f"  RL Performance: {rl_perf:.2f}/step")
print(f"  Rank: {rl_rank}/{len(results)}")
print(f"  Best Baseline: {best_baseline:.2f}/step ({best_baseline_name})")
print(f"  Difference: {rl_perf - best_baseline:+.2f} ({((rl_perf - best_baseline)/best_baseline)*100:+.1f}%)")

print("\n" + "=" * 70)

if rl_perf > best_baseline:
    margin = ((rl_perf - best_baseline) / best_baseline) * 100
    print("✅ RL BEATS ALL BASELINES!")
    print(f"   Margin: +{margin:.1f}% over best baseline")
    print("\n🎊 THESIS CONCLUSION:")
    print("   'Optimized Q-Learning with random initialization")
    print("   successfully outperforms all fixed-time baselines,")
    print(f"   achieving {rl_perf:.2f}/step vs best baseline {best_baseline:.2f}/step'")
elif rl_perf > results['30s baseline']['per_step']:
    print("✅ RL BEATS ORIGINAL 30s BASELINE!")
    margin = ((rl_perf - results['30s baseline']['per_step']) / results['30s baseline']['per_step']) * 100
    print(f"   Margin: +{margin:.1f}%")
else:
    print("⚠️  RL competitive but below optimal baseline")

print("=" * 70)

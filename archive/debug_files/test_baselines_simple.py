import sys
sys.path.append('src')
import numpy as np
import json

# Import environment
from simulation.environment import JammingMachine

print("=" * 70)
print("TESTING MULTIPLE FIXED-TIME BASELINES")
print("=" * 70)

with open('config/sim_config.json') as f:
    config = json.load(f)

# Test different intervals
intervals = [15, 20, 25, 30, 35, 40, 45]
results = {}

for interval in intervals:
    print(f"\nTesting {interval}-second interval...")
    env = JammingMachine(config)
    
    rewards = []
    for episode in range(10):  # 10 episodes for speed
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            # Fixed-time logic
            cycle_position = step % (interval * 2)
            if cycle_position < interval:
                action = 0  # NS green
            else:
                action = 3  # EW green
            
            state, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1
        
        rewards.append(episode_reward)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    results[interval] = {
        'mean': mean_reward,
        'std': std_reward,
        'per_step': mean_reward / 300
    }
    
    print(f"  Mean: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Per step: {mean_reward/300:.2f}")

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

best_interval = max(results.keys(), key=lambda k: results[k]['mean'])
worst_interval = min(results.keys(), key=lambda k: results[k]['mean'])

print(f"\n{'Interval':<10} {'Reward/Step':<15} {'Status':<20}")
print("-" * 70)

for interval in sorted(results.keys()):
    r = results[interval]
    
    if interval == best_interval:
        marker = "← BEST"
    elif interval == worst_interval:
        marker = "← WORST"
    elif interval == 30:
        marker = "← CURRENT"
    else:
        marker = ""
    
    print(f"{interval}s{' '*7} {r['per_step']:.2f}{' '*11} {marker}")

print("\n" + "=" * 70)
print("KEY FINDINGS:")
print("=" * 70)

best_perf = results[best_interval]['per_step']
current_perf = results[30]['per_step']
worst_perf = results[worst_interval]['per_step']

print(f"Best: {best_interval}s → {best_perf:.2f}/step")
print(f"Current: 30s → {current_perf:.2f}/step")
print(f"Worst: {worst_interval}s → {worst_perf:.2f}/step")
print(f"\nRange: {worst_perf:.2f} to {best_perf:.2f}")
print(f"Difference: {best_perf - worst_perf:.2f}")

# Analysis
diff_from_best = best_perf - current_perf

if diff_from_best < 0.5:
    print("\n" + "🎯 " + "="*65)
    print("CONCLUSION: 30s is NEAR-OPTIMAL or THE BEST baseline!")
    print("="*70)
    print("\nThis means:")
    print("  ✅ Your baseline comparison is FAIR")
    print("  ✅ RL truly struggling against good fixed-time")
    print("  ✅ This is a valid research finding")
    print("\nRecommendations:")
    print("  1. Accept current results (honest analysis)")
    print("  2. Test different traffic scenarios (where RL might win)")
    print("  3. Try DQN (for comprehensive comparison)")
    
elif diff_from_best < 2.0:
    print("\n" + "⚠️  " + "="*65)
    print("CONCLUSION: 30s is GOOD but not optimal")
    print("="*70)
    print(f"\n{best_interval}s performs {diff_from_best:.2f} better per step")
    print("\nThis means:")
    print("  • You can show RL vs multiple baselines")
    print("  • RL might beat some (weaker) baselines")
    print(f"  • {best_interval}s is the 'fair' comparison point")
    
else:
    print("\n" + "✅ " + "="*65)
    print("CONCLUSION: 30s is SUBOPTIMAL baseline!")
    print("="*70)
    print(f"\n{best_interval}s performs {diff_from_best:.2f} better per step")
    print("\nThis means:")
    print("  • Your RL might actually be competitive!")
    print(f"  • Should re-evaluate vs {best_interval}s baseline")
    print("  • Current comparison used weak baseline")

# RL comparison
rl_perf = 38.34  # From your Exp 3
print("\n" + "=" * 70)
print("RL PERFORMANCE CONTEXT")
print("=" * 70)
print(f"RL (Exp 3): {rl_perf:.2f}/step")
print(f"Best baseline: {best_perf:.2f}/step")
print(f"Difference: {rl_perf - best_perf:.2f} ({((rl_perf - best_perf)/best_perf)*100:+.1f}%)")

baselines_beaten = sum(1 for r in results.values() if rl_perf > r['per_step'])
print(f"\nRL beats {baselines_beaten}/{len(results)} tested baselines")

if baselines_beaten > 0:
    print(f"✅ RL beats: {', '.join([f'{k}s' for k, v in results.items() if rl_perf > v['per_step']])}")

print("=" * 70)

#!/bin/bash
# Ablation Study Setup Script
cd /mnt/project

# Create directories
mkdir -p scripts results backup

# Create ablation_1_full_system.py
cat > scripts/ablation_1_full_system.py << 'SCRIPT1'
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from environment import JammingMachine
from q_learning_agent import QLearningAgent

print("="*70)
print("ABLATION 1: FULL SYSTEM")
print("="*70)

with open('sim_config.json') as f: sim_config = json.load(f)
with open('initial_state.json') as f: initial_state = json.load(f)
for lane in initial_state:
    if 'vehicle_count' in initial_state[lane]:
        initial_state[lane]['vehicle_count'] //= 3

env = JammingMachine(sim_config, initial_state)
agent = QLearningAgent(8, 6, 0.0, 0.95, 0.0)
agent.load_q_table('trained_q_table.json')

results = {'total_rewards': [], 'total_waiting_times': [], 'avg_queue_lengths': [], 'throughputs': []}
for ep in range(100):
    state, done, ep_reward, ep_wait = env.reset(), False, 0, 0
    while not done:
        action = agent.choose_action(state)
        state, reward, done, info = env.step(action)
        ep_reward += reward
        ep_wait += sum(env.waiting_times.values())
    results['total_rewards'].append(ep_reward)
    results['total_waiting_times'].append(ep_wait)
    results['avg_queue_lengths'].append(np.mean(list(env.vehicle_counts.values())))
    results['throughputs'].append(info.get('throughput', 0))
    if (ep + 1) % 20 == 0: print(f"  Episode {ep+1}/100 | Avg: {np.mean(results['total_rewards'][-20:]):.2f}")

summary = {'configuration': 'full_system', 'episodes': 100,
           'mean_reward': float(np.mean(results['total_rewards'])),
           'std_reward': float(np.std(results['total_rewards'])),
           'detailed_results': results}
with open('results/ablation_1_full_system.json', 'w') as f: json.dump(summary, f, indent=2)
print(f"\n✅ Mean Reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
SCRIPT1

# Create ablation_2_without_yolo.py
cat > scripts/ablation_2_without_yolo.py << 'SCRIPT2'
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from environment import JammingMachine
from q_learning_agent import QLearningAgent

print("="*70)
print("ABLATION 2: WITHOUT YOLO")
print("="*70)

with open('sim_config.json') as f: sim_config = json.load(f)
np.random.seed(42)
random_init = {f'lane_{d}': {'vehicle_count': np.random.randint(3,8), 'density_score': np.random.uniform(0.2,0.5)} 
               for d in ['north','south','east','west']}

env = JammingMachine(sim_config, random_init)
agent = QLearningAgent(8, 6, 0.0, 0.95, 0.0)
agent.load_q_table('trained_q_table.json')

results = {'total_rewards': [], 'total_waiting_times': [], 'avg_queue_lengths': [], 'throughputs': []}
for ep in range(100):
    for lane in random_init:
        random_init[lane]['vehicle_count'] = np.random.randint(3,8)
        random_init[lane]['density_score'] = np.random.uniform(0.2,0.5)
    state, done, ep_reward, ep_wait = env.reset(), False, 0, 0
    while not done:
        action = agent.choose_action(state)
        state, reward, done, info = env.step(action)
        ep_reward += reward
        ep_wait += sum(env.waiting_times.values())
    results['total_rewards'].append(ep_reward)
    results['total_waiting_times'].append(ep_wait)
    results['avg_queue_lengths'].append(np.mean(list(env.vehicle_counts.values())))
    results['throughputs'].append(info.get('throughput', 0))
    if (ep + 1) % 20 == 0: print(f"  Episode {ep+1}/100 | Avg: {np.mean(results['total_rewards'][-20:]):.2f}")

summary = {'configuration': 'without_yolo', 'episodes': 100,
           'mean_reward': float(np.mean(results['total_rewards'])),
           'std_reward': float(np.std(results['total_rewards'])),
           'detailed_results': results}
with open('results/ablation_2_without_yolo.json', 'w') as f: json.dump(summary, f, indent=2)
print(f"\n✅ Mean Reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
SCRIPT2

# Create ablation_3_without_dbscan.py
cat > scripts/ablation_3_without_dbscan.py << 'SCRIPT3'
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from environment import JammingMachine
from q_learning_agent import QLearningAgent

print("="*70)
print("ABLATION 3: WITHOUT DBSCAN")
print("="*70)

with open('sim_config.json') as f: sim_config = json.load(f)
with open('initial_state.json') as f: initial_state = json.load(f)
for lane in initial_state:
    if 'vehicle_count' in initial_state[lane]: initial_state[lane]['vehicle_count'] //= 3
    initial_state[lane]['density_score'] = 0.0

env = JammingMachine(sim_config, initial_state)
agent = QLearningAgent(8, 6, 0.0, 0.95, 0.0)
agent.load_q_table('trained_q_table.json')

results = {'total_rewards': [], 'total_waiting_times': [], 'avg_queue_lengths': [], 'throughputs': []}
for ep in range(100):
    state, done, ep_reward, ep_wait = env.reset(), False, 0, 0
    while not done:
        action = agent.choose_action(state)
        state, reward, done, info = env.step(action)
        ep_reward += reward
        ep_wait += sum(env.waiting_times.values())
    results['total_rewards'].append(ep_reward)
    results['total_waiting_times'].append(ep_wait)
    results['avg_queue_lengths'].append(np.mean(list(env.vehicle_counts.values())))
    results['throughputs'].append(info.get('throughput', 0))
    if (ep + 1) % 20 == 0: print(f"  Episode {ep+1}/100 | Avg: {np.mean(results['total_rewards'][-20:]):.2f}")

summary = {'configuration': 'without_dbscan', 'episodes': 100,
           'mean_reward': float(np.mean(results['total_rewards'])),
           'std_reward': float(np.std(results['total_rewards'])),
           'detailed_results': results}
with open('results/ablation_3_without_dbscan.json', 'w') as f: json.dump(summary, f, indent=2)
print(f"\n✅ Mean Reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
SCRIPT3

# Create ablation_4_without_rl.py
cat > scripts/ablation_4_without_rl.py << 'SCRIPT4'
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from environment import JammingMachine

print("="*70)
print("ABLATION 4: WITHOUT RL")
print("="*70)

class FixedAgent:
    def __init__(self): self.seq, self.step, self.idx = [1,4], 0, 0
    def choose_action(self, state):
        if self.step > 0 and self.step % 30 == 0: self.idx = (self.idx + 1) % 2
        self.step += 1; return self.seq[self.idx]
    def reset(self): self.step, self.idx = 0, 0

with open('sim_config.json') as f: sim_config = json.load(f)
with open('initial_state.json') as f: initial_state = json.load(f)
for lane in initial_state:
    if 'vehicle_count' in initial_state[lane]: initial_state[lane]['vehicle_count'] //= 3

env = JammingMachine(sim_config, initial_state)
agent = FixedAgent()

results = {'total_rewards': [], 'total_waiting_times': [], 'avg_queue_lengths': [], 'throughputs': []}
for ep in range(100):
    agent.reset()
    state, done, ep_reward, ep_wait = env.reset(), False, 0, 0
    while not done:
        action = agent.choose_action(state)
        state, reward, done, info = env.step(action)
        ep_reward += reward
        ep_wait += sum(env.waiting_times.values())
    results['total_rewards'].append(ep_reward)
    results['total_waiting_times'].append(ep_wait)
    results['avg_queue_lengths'].append(np.mean(list(env.vehicle_counts.values())))
    results['throughputs'].append(info.get('throughput', 0))
    if (ep + 1) % 20 == 0: print(f"  Episode {ep+1}/100 | Avg: {np.mean(results['total_rewards'][-20:]):.2f}")

summary = {'configuration': 'without_rl', 'episodes': 100,
           'mean_reward': float(np.mean(results['total_rewards'])),
           'std_reward': float(np.std(results['total_rewards'])),
           'detailed_results': results}
with open('results/ablation_4_without_rl.json', 'w') as f: json.dump(summary, f, indent=2)
print(f"\n✅ Mean Reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
SCRIPT4

# Create analyze_ablation.py
cat > scripts/analyze_ablation.py << 'SCRIPT5'
import json, numpy as np
from scipy import stats

configs = {'Full System': 'results/ablation_1_full_system.json',
           'Without YOLO': 'results/ablation_2_without_yolo.json',
           'Without DBSCAN': 'results/ablation_3_without_dbscan.json',
           'Without RL': 'results/ablation_4_without_rl.json'}

results = {}
for name, path in configs.items():
    with open(path) as f: results[name] = json.load(f)

print("\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)
for name in configs.keys():
    r = results[name]
    print(f"{name:<20} {r['mean_reward']:>8.2f} ± {r['std_reward']:<6.2f}")

baseline = results['Full System']['mean_reward']
print("\n" + "="*80)
print("COMPONENT CONTRIBUTIONS")
print("="*80)
for comp, key in [('YOLO', 'Without YOLO'), ('DBSCAN', 'Without DBSCAN'), ('RL', 'Without RL')]:
    deg = ((baseline - results[key]['mean_reward']) / baseline) * 100
    print(f"{comp:<10} {deg:>6.2f}% {'🔴 CRITICAL' if deg > 20 else '⚠️ IMPORTANT' if deg > 5 else '✓'}")

print("\n" + "="*80)
print("STATISTICAL TESTS (vs Full System)")
print("="*80)
full_r = np.array(results['Full System']['detailed_results']['total_rewards'])
for name in ['Without YOLO', 'Without DBSCAN', 'Without RL']:
    config_r = np.array(results[name]['detailed_results']['total_rewards'])
    t, p = stats.ttest_ind(full_r, config_r)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"{name:<20} t={t:>7.2f}  p={p:.6f} {sig}")
print("\n✅ Ablation Analysis Complete!")
SCRIPT5

# Create master runner
cat > run_ablation_study.py << 'MASTER'
import subprocess, sys
scripts = ["scripts/ablation_1_full_system.py", "scripts/ablation_2_without_yolo.py",
           "scripts/ablation_3_without_dbscan.py", "scripts/ablation_4_without_rl.py"]
print("🚀 Running Ablation Study (400 episodes)...\n")
for s in scripts:
    print(f"Running {s}...")
    subprocess.run([sys.executable, s])
print("\n✅ Done! Now run: python3 scripts/analyze_ablation.py")
MASTER

chmod +x scripts/*.py run_ablation_study.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run:"
echo "  cd /mnt/project"
echo "  python3 run_ablation_study.py"
echo "  python3 scripts/analyze_ablation.py"
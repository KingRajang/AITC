#!/usr/bin/env python3
"""
MASTER SETUP SCRIPT
Generates all experiment scripts in one go - no downloads needed!

Usage:
    python setup_experiments.py
"""

import os

def create_file(path, content):
    """Create a file with content"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    print(f"✓ Created: {path}")

def main():
    print("="*60)
    print("🚀 SETTING UP EXPERIMENT SCRIPTS")
    print("="*60)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # 1. ACTUATED BASELINE
    actuated = '''#!/usr/bin/env python3
"""Actuated Traffic Controller Baseline"""
import json
import numpy as np
from src.simulation.environment import JammingMachine

class ActuatedController:
    def __init__(self):
        self.current_phase = "NS"
    
    def choose_action(self, env):
        ns_demand = env.vehicle_counts['North'] + env.vehicle_counts['South']
        ew_demand = env.vehicle_counts['East'] + env.vehicle_counts['West']
        
        if self.current_phase == "NS":
            if ns_demand < 5:
                action = 0  # Short
            elif ns_demand < 10:
                action = 1  # Normal
            else:
                action = 2  # Long
        else:
            if ew_demand < 5:
                action = 3
            elif ew_demand < 10:
                action = 4
            else:
                action = 5
        
        self.current_phase = "EW" if action < 3 else "NS"
        return action

def run_evaluation(episodes=100):
    print("Evaluating Actuated Controller...")
    
    with open('config/sim_config.json') as f:
        sim_config = json.load(f)
    
    try:
        with open('data/initial_state.json') as f:
            initial_state = json.load(f)
    except:
        initial_state = None
    
    if initial_state:
        for lane_id in initial_state:
            initial_state[lane_id]['vehicle_count'] //= 3
    
    env = JammingMachine(sim_config, initial_state)
    controller = ActuatedController()
    
    results = {'total_rewards': [], 'avg_queue_lengths': []}
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = controller.choose_action(env)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        
        results['total_rewards'].append(total_reward)
        results['avg_queue_lengths'].append(sum(env.vehicle_counts.values()) / 4)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.2f}")
    
    summary = {
        'controller': 'actuated',
        'mean_reward': float(np.mean(results['total_rewards'])),
        'std_reward': float(np.std(results['total_rewards'])),
        'results': results
    }
    
    with open('results/actuated_baseline.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\\nActuated Results: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    return summary

if __name__ == "__main__":
    run_evaluation()
'''
    create_file('scripts/actuated_baseline.py', actuated)
    
    # 2. FIXED-TIME BASELINE
    fixed = '''#!/usr/bin/env python3
"""Fixed-Time Traffic Controller Baseline"""
import json
import numpy as np
from src.simulation.environment import JammingMachine

class FixedTimeController:
    def __init__(self):
        self.step = 0
        self.cycle = 60  # 30s NS + 30s EW
    
    def choose_action(self):
        action = 1 if self.step < 30 else 4  # NS Normal or EW Normal
        self.step = (self.step + 1) % self.cycle
        return action

def run_evaluation(episodes=100):
    print("Evaluating Fixed-Time Controller...")
    
    with open('config/sim_config.json') as f:
        sim_config = json.load(f)
    
    try:
        with open('data/initial_state.json') as f:
            initial_state = json.load(f)
    except:
        initial_state = None
    
    if initial_state:
        for lane_id in initial_state:
            initial_state[lane_id]['vehicle_count'] //= 3
    
    env = JammingMachine(sim_config, initial_state)
    controller = FixedTimeController()
    
    results = {'total_rewards': [], 'avg_queue_lengths': []}
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        controller.step = 0
        
        while not done:
            action = controller.choose_action()
            state, reward, done, _ = env.step(action)
            total_reward += reward
        
        results['total_rewards'].append(total_reward)
        results['avg_queue_lengths'].append(sum(env.vehicle_counts.values()) / 4)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.2f}")
    
    summary = {
        'controller': 'fixed_time',
        'mean_reward': float(np.mean(results['total_rewards'])),
        'std_reward': float(np.std(results['total_rewards'])),
        'results': results
    }
    
    with open('results/fixed_time_baseline.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\\nFixed-Time Results: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    return summary

if __name__ == "__main__":
    run_evaluation()
'''
    create_file('scripts/fixed_time_baseline.py', fixed)
    
    # 3. Q-LEARNING EVALUATION
    ql_eval = '''#!/usr/bin/env python3
"""Q-Learning Agent Evaluation"""
import json
import numpy as np
from src.simulation.environment import JammingMachine
from src.agent.q_learning_agent import QLearningAgent

def run_evaluation(episodes=100):
    print("Evaluating Q-Learning Agent...")
    
    with open('config/sim_config.json') as f:
        sim_config = json.load(f)
    
    try:
        with open('data/initial_state.json') as f:
            initial_state = json.load(f)
    except:
        initial_state = None
    
    if initial_state:
        for lane_id in initial_state:
            initial_state[lane_id]['vehicle_count'] //= 3
    
    env = JammingMachine(sim_config, initial_state)
    agent = QLearningAgent(8, 6, 0.1, 0.95, 0.0)  # epsilon=0 for eval
    agent.load_q_table('trained_q_table.json')
    
    results = {'total_rewards': [], 'avg_queue_lengths': []}
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        
        results['total_rewards'].append(total_reward)
        results['avg_queue_lengths'].append(sum(env.vehicle_counts.values()) / 4)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.2f}")
    
    summary = {
        'controller': 'q_learning',
        'mean_reward': float(np.mean(results['total_rewards'])),
        'std_reward': float(np.std(results['total_rewards'])),
        'results': results
    }
    
    with open('results/q_learning_evaluation.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\\nQ-Learning Results: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    return summary

if __name__ == "__main__":
    run_evaluation()
'''
    create_file('scripts/evaluate_ql.py', ql_eval)
    
    # 4. STATISTICAL COMPARISON
    stats = '''#!/usr/bin/env python3
"""Statistical Comparison"""
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def compare_results():
    print("\\nStatistical Comparison")
    print("="*60)
    
    # Load results
    with open('results/q_learning_evaluation.json') as f:
        ql = json.load(f)
    with open('results/actuated_baseline.json') as f:
        act = json.load(f)
    with open('results/fixed_time_baseline.json') as f:
        fix = json.load(f)
    
    ql_rewards = ql['results']['total_rewards']
    act_rewards = act['results']['total_rewards']
    fix_rewards = fix['results']['total_rewards']
    
    # Statistical tests
    print("\\n1. Q-Learning vs Actuated")
    t_stat, p_val = stats.ttest_ind(ql_rewards, act_rewards)
    improvement = ((np.mean(ql_rewards) - np.mean(act_rewards)) / np.mean(act_rewards)) * 100
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value: {p_val:.6f}")
    print(f"   Improvement: {improvement:.2f}%")
    
    print("\\n2. Q-Learning vs Fixed-Time")
    t_stat, p_val = stats.ttest_ind(ql_rewards, fix_rewards)
    improvement = ((np.mean(ql_rewards) - np.mean(fix_rewards)) / np.mean(fix_rewards)) * 100
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value: {p_val:.6f}")
    print(f"   Improvement: {improvement:.2f}%")
    
    # Summary table
    print("\\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Controller':<15} {'Mean Reward':<15} {'Std Dev':<10}")
    print("-"*60)
    print(f"{'Q-Learning':<15} {ql['mean_reward']:<15.2f} {ql['std_reward']:<10.2f}")
    print(f"{'Actuated':<15} {act['mean_reward']:<15.2f} {act['std_reward']:<10.2f}")
    print(f"{'Fixed-Time':<15} {fix['mean_reward']:<15.2f} {fix['std_reward']:<10.2f}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    controllers = ['Q-Learning\\n(Ours)', 'Actuated', 'Fixed-Time']
    means = [ql['mean_reward'], act['mean_reward'], fix['mean_reward']]
    stds = [ql['std_reward'], act['std_reward'], fix['std_reward']]
    
    ax.bar(controllers, means, yerr=stds, capsize=10, 
           color=['#2E7D32', '#1976D2', '#D32F2F'], alpha=0.7)
    ax.set_ylabel('Mean Total Reward', fontsize=12)
    ax.set_title('Controller Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/comparison_plot.png', dpi=300)
    print("\\n✓ Plot saved to: results/comparison_plot.png")

if __name__ == "__main__":
    compare_results()
'''
    create_file('scripts/compare_results.py', stats)
    
    # 5. MASTER RUNNER
    runner = '''#!/usr/bin/env python3
"""Run All Experiments"""
import os
import sys

def run(cmd, desc):
    print(f"\\n{'='*60}")
    print(f"⏳ {desc}")
    print("="*60)
    result = os.system(cmd)
    if result != 0:
        print(f"❌ FAILED: {desc}")
        sys.exit(1)
    print(f"✅ DONE: {desc}")

print("🚀 RUNNING ALL EXPERIMENTS")
print("="*60)

# 1. Train Q-Learning (if not already trained)
if not os.path.exists('trained_q_table.json'):
    run('python scripts/main_rl_training.py', '1. Training Q-Learning (10K episodes)')
else:
    print("\\n✓ Q-Learning already trained (trained_q_table.json exists)")

# 2. Evaluate all controllers
run('python scripts/evaluate_ql.py', '2. Evaluating Q-Learning (100 episodes)')
run('python scripts/actuated_baseline.py', '3. Running Actuated Baseline')
run('python scripts/fixed_time_baseline.py', '4. Running Fixed-Time Baseline')

# 3. Compare results
run('python scripts/compare_results.py', '5. Statistical Comparison')

print("\\n" + "="*60)
print("🎉 ALL EXPERIMENTS COMPLETE!")
print("="*60)
print("\\nResults:")
print("  ├── results/q_learning_evaluation.json")
print("  ├── results/actuated_baseline.json")
print("  ├── results/fixed_time_baseline.json")
print("  └── results/comparison_plot.png")
print("\\nNext: Check results/ folder for all outputs")
'''
    create_file('run_experiments.py', runner)
    
    # 6. QUICK README
    readme = '''# QUICK START GUIDE

## 1️⃣ Setup (Already Done!)
You just ran setup_experiments.py ✓

## 2️⃣ Run Experiments

### Option A: All at once (Recommended)
```bash
python run_experiments.py
```
This will:
- Train Q-Learning (if not done)
- Evaluate all 3 controllers
- Generate comparison statistics
- Create comparison plot

**Time:** 4-5 hours total

### Option B: Step by step
```bash
# Train Q-Learning
python scripts/main_rl_training.py

# Evaluate all controllers
python scripts/evaluate_ql.py
python scripts/actuated_baseline.py
python scripts/fixed_time_baseline.py

# Compare results
python scripts/compare_results.py
```

## 3️⃣ Check Results
```bash
# View summary
cat results/q_learning_evaluation.json
cat results/actuated_baseline.json
cat results/fixed_time_baseline.json

# View plot
open results/comparison_plot.png
```

## 4️⃣ For Paper
Results are in `results/` folder:
- Statistical comparison printed to console
- Plot ready for paper in `results/comparison_plot.png`
- Raw data in JSON files

## 🚨 Troubleshooting

**ImportError?**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Files not found?**
Make sure you're in the project root directory

**Still stuck?**
Run step by step (Option B) to identify the issue
'''
    create_file('EXPERIMENTS_README.md', readme)
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    print("\n📁 Created:")
    print("  ├── scripts/actuated_baseline.py")
    print("  ├── scripts/fixed_time_baseline.py")
    print("  ├── scripts/evaluate_ql.py")
    print("  ├── scripts/compare_results.py")
    print("  ├── run_experiments.py")
    print("  └── EXPERIMENTS_README.md")
    print("\n🚀 NEXT STEPS:")
    print("  1. Read: EXPERIMENTS_README.md")
    print("  2. Run: python run_experiments.py")
    print("  3. Wait: ~4-5 hours for results")
    print("  4. Check: results/ folder")

if __name__ == "__main__":
    main()
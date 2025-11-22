#!/usr/bin/env python3
"""Run All Experiments"""
import os
import sys

def run(cmd, desc):
    print(f"\n{'='*60}")
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
    print("\n✓ Q-Learning already trained (trained_q_table.json exists)")

# 2. Evaluate all controllers
run('python scripts/evaluate_ql.py', '2. Evaluating Q-Learning (100 episodes)')
run('python scripts/actuated_baseline.py', '3. Running Actuated Baseline')
run('python scripts/fixed_time_baseline.py', '4. Running Fixed-Time Baseline')

# 3. Compare results
run('python scripts/compare_results.py', '5. Statistical Comparison')

print("\n" + "="*60)
print("🎉 ALL EXPERIMENTS COMPLETE!")
print("="*60)
print("\nResults:")
print("  ├── results/q_learning_evaluation.json")
print("  ├── results/actuated_baseline.json")
print("  ├── results/fixed_time_baseline.json")
print("  └── results/comparison_plot.png")
print("\nNext: Check results/ folder for all outputs")

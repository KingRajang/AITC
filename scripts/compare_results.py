#!/usr/bin/env python3
"""Statistical Comparison"""
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def compare_results():
    print("\nStatistical Comparison")
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
    print("\n1. Q-Learning vs Actuated")
    t_stat, p_val = stats.ttest_ind(ql_rewards, act_rewards)
    improvement = ((np.mean(ql_rewards) - np.mean(act_rewards)) / np.mean(act_rewards)) * 100
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value: {p_val:.6f}")
    print(f"   Improvement: {improvement:.2f}%")
    
    print("\n2. Q-Learning vs Fixed-Time")
    t_stat, p_val = stats.ttest_ind(ql_rewards, fix_rewards)
    improvement = ((np.mean(ql_rewards) - np.mean(fix_rewards)) / np.mean(fix_rewards)) * 100
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value: {p_val:.6f}")
    print(f"   Improvement: {improvement:.2f}%")
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Controller':<15} {'Mean Reward':<15} {'Std Dev':<10}")
    print("-"*60)
    print(f"{'Q-Learning':<15} {ql['mean_reward']:<15.2f} {ql['std_reward']:<10.2f}")
    print(f"{'Actuated':<15} {act['mean_reward']:<15.2f} {act['std_reward']:<10.2f}")
    print(f"{'Fixed-Time':<15} {fix['mean_reward']:<15.2f} {fix['std_reward']:<10.2f}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    controllers = ['Q-Learning\n(Ours)', 'Actuated', 'Fixed-Time']
    means = [ql['mean_reward'], act['mean_reward'], fix['mean_reward']]
    stds = [ql['std_reward'], act['std_reward'], fix['std_reward']]
    
    ax.bar(controllers, means, yerr=stds, capsize=10, 
           color=['#2E7D32', '#1976D2', '#D32F2F'], alpha=0.7)
    ax.set_ylabel('Mean Total Reward', fontsize=12)
    ax.set_title('Controller Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/comparison_plot.png', dpi=300)
    print("\n✓ Plot saved to: results/comparison_plot.png")

if __name__ == "__main__":
    compare_results()

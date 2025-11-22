#!/usr/bin/env python3
"""
Systematic Optimization Experiments
Goal: Beat Actuated Control (21,844 reward)
Current: 21,789 reward
Gap: 55 points (0.25%)

Strategy: Test 5 configurations systematically
"""

import json
import os
import shutil
from datetime import datetime

# Target to beat
ACTUATED_BASELINE = 21844.29

# Configurations to test
EXPERIMENTS = [
    {
        "id": "EXP-001",
        "name": "Baseline (Current)",
        "description": "Current configuration for reference",
        "reward_weights": {
            "throughput": 0.40,
            "queue": 0.30,
            "waiting": 0.20,
            "fairness": 0.10
        },
        "skip_training": True  # Use existing trained_q_table.json
    },
    {
        "id": "EXP-002",
        "name": "Throughput Emphasis",
        "description": "Increase throughput weight, reduce fairness",
        "reward_weights": {
            "throughput": 0.50,
            "queue": 0.25,
            "waiting": 0.20,
            "fairness": 0.05
        },
        "skip_training": False
    },
    {
        "id": "EXP-003",
        "name": "Queue Minimization",
        "description": "Focus on reducing queue lengths",
        "reward_weights": {
            "throughput": 0.35,
            "queue": 0.40,
            "waiting": 0.20,
            "fairness": 0.05
        },
        "skip_training": False
    },
    {
        "id": "EXP-004",
        "name": "Balanced Premium",
        "description": "Slight boost to throughput and queue",
        "reward_weights": {
            "throughput": 0.45,
            "queue": 0.35,
            "waiting": 0.15,
            "fairness": 0.05
        },
        "skip_training": False
    },
    {
        "id": "EXP-005",
        "name": "Throughput-Queue Focus",
        "description": "Maximize flow while minimizing queues",
        "reward_weights": {
            "throughput": 0.50,
            "queue": 0.30,
            "waiting": 0.15,
            "fairness": 0.05
        },
        "skip_training": False
    }
]

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_experiment_info(exp):
    """Print experiment details"""
    print(f"\n📋 Experiment: {exp['id']} - {exp['name']}")
    print(f"   {exp['description']}")
    print(f"\n   Reward Weights:")
    for key, val in exp['reward_weights'].items():
        print(f"      {key:12s}: {val:.2f}")

def update_config(reward_weights):
    """Update sim_config.json with new weights"""
    config_path = 'sim_config.json'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['reward_weights'] = reward_weights
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"   ✓ Updated {config_path}")

def backup_model():
    """Backup existing trained model"""
    if os.path.exists('trained_q_table.json'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f'trained_q_table_backup_{timestamp}.json'
        shutil.copy('trained_q_table.json', backup_name)
        print(f"   ✓ Backed up model to {backup_name}")
        return backup_name
    return None

def run_training():
    """Run training script"""
    print(f"\n   ⏳ Training Q-Learning agent (this will take 20-40 minutes)...")
    print(f"   💡 You can monitor progress in the terminal output")
    
    import subprocess
    result = subprocess.run(
        ['python', 'main_rl_training.py'],
        capture_output=False,  # Show live output
        text=True
    )
    
    if result.returncode == 0:
        print(f"   ✓ Training completed successfully")
        return True
    else:
        print(f"   ✗ Training failed")
        return False

def run_evaluation():
    """Run evaluation and return results"""
    print(f"\n   ⏳ Evaluating trained agent (500 episodes)...")
    
    import subprocess
    result = subprocess.run(
        ['python', 'run_experiments.py'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"   ✗ Evaluation failed")
        print(result.stderr)
        return None
    
    # Parse results
    try:
        with open('results/q_learning_evaluation.json', 'r') as f:
            results = json.load(f)
        
        mean_reward = results['mean_reward']
        std_reward = results['std_reward']
        
        print(f"   ✓ Evaluation completed")
        return mean_reward, std_reward
    except Exception as e:
        print(f"   ✗ Could not parse results: {e}")
        return None

def save_experiment_result(exp, reward, std, backup_file):
    """Save experiment result to log"""
    result = {
        'timestamp': datetime.now().isoformat(),
        'experiment_id': exp['id'],
        'experiment_name': exp['name'],
        'description': exp['description'],
        'reward_weights': exp['reward_weights'],
        'mean_reward': reward,
        'std_reward': std,
        'vs_actuated': reward - ACTUATED_BASELINE,
        'vs_actuated_percent': ((reward - ACTUATED_BASELINE) / ACTUATED_BASELINE) * 100,
        'beats_actuated': reward > ACTUATED_BASELINE,
        'model_backup': backup_file
    }
    
    # Save individual result
    result_file = f'results/optimization_{exp["id"]}.json'
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

def print_result(exp, reward, std):
    """Print experiment result"""
    diff = reward - ACTUATED_BASELINE
    pct = (diff / ACTUATED_BASELINE) * 100
    
    print(f"\n   📊 RESULTS:")
    print(f"      Mean Reward:  {reward:.2f} ± {std:.2f}")
    print(f"      vs Actuated:  {diff:+.2f} ({pct:+.2f}%)")
    
    if reward > ACTUATED_BASELINE:
        print(f"      🎉 BEATS ACTUATED! ✓")
    elif abs(diff) < 30:  # Within margin
        print(f"      📊 Very close to Actuated")
    else:
        print(f"      📉 Below Actuated")

def print_summary(results):
    """Print final summary"""
    print_header("🏆 OPTIMIZATION SUMMARY")
    
    # Sort by reward
    sorted_results = sorted(results, key=lambda x: x['mean_reward'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Experiment':<25} {'Reward':<15} {'vs Actuated':<15} {'Status':<10}")
    print("-"*80)
    
    for i, result in enumerate(sorted_results, 1):
        exp_name = result['experiment_name'][:24]
        reward = result['mean_reward']
        vs_act = result['vs_actuated']
        status = "✓ WINS" if result['beats_actuated'] else "≈ CLOSE" if abs(vs_act) < 30 else "✗ Lower"
        
        print(f"{i:<6} {exp_name:<25} {reward:<15.2f} {vs_act:+15.2f} {status:<10}")
    
    # Best result
    best = sorted_results[0]
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   {best['experiment_name']}")
    print(f"   Reward: {best['mean_reward']:.2f} ± {best['std_reward']:.2f}")
    print(f"   vs Actuated: {best['vs_actuated']:+.2f} ({best['vs_actuated_percent']:+.2f}%)")
    print(f"\n   Reward Weights:")
    for key, val in best['reward_weights'].items():
        print(f"      {key:12s}: {val:.2f}")
    
    if best['beats_actuated']:
        print(f"\n   🎉 CONGRATULATIONS! You beat Actuated Control!")
        print(f"   💡 Use this configuration for your final thesis results")
        print(f"   📁 Model backed up as: {best['model_backup']}")
    else:
        print(f"\n   📊 Close result - consider:")
        print(f"      1. Using best config for final results")
        print(f"      2. Framing as 'competitive with actuated'")
        print(f"      3. Trying 1-2 more targeted experiments")
    
    # Save summary
    summary_file = 'results/optimization_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'optimization_date': datetime.now().isoformat(),
            'target_baseline': ACTUATED_BASELINE,
            'experiments': sorted_results
        }, f, indent=2)
    
    print(f"\n   ✓ Summary saved to: {summary_file}")

def main():
    """Run all optimization experiments"""
    print_header("🚀 SYSTEMATIC OPTIMIZATION EXPERIMENTS")
    
    print(f"\n📌 Current Status:")
    print(f"   Q-Learning Baseline: 21,789 ± 45")
    print(f"   Actuated Control:    21,844 ± 42  ← Target")
    print(f"   Gap to close:        55 points (0.25%)")
    
    print(f"\n📋 Experiments to Run: {len(EXPERIMENTS)}")
    print(f"   1 baseline (using existing model)")
    print(f"   {len(EXPERIMENTS)-1} optimization attempts")
    
    print(f"\n⏱️  Estimated Time:")
    print(f"   Training: ~30 min per experiment × {len(EXPERIMENTS)-1} = ~2 hours")
    print(f"   Evaluation: ~5 min per experiment × {len(EXPERIMENTS)} = ~25 min")
    print(f"   TOTAL: ~2.5 hours")
    
    print(f"\n💡 Strategy:")
    print(f"   1. Test baseline with existing model (reference)")
    print(f"   2. Try throughput-focused configurations")
    print(f"   3. Try queue-minimization approach")
    print(f"   4. Compare all results")
    
    input(f"\n   Press ENTER to start, or Ctrl+C to cancel...")
    
    results = []
    
    for exp in EXPERIMENTS:
        print_header(f"Experiment {exp['id']}: {exp['name']}")
        print_experiment_info(exp)
        
        # Update configuration
        update_config(exp['reward_weights'])
        
        # Training
        if exp['skip_training']:
            print(f"\n   ⏭️  Skipping training (using existing model)")
            backup_file = None
        else:
            backup_file = backup_model()
            
            # Remove old model to force retraining
            if os.path.exists('trained_q_table.json'):
                os.remove('trained_q_table.json')
                print(f"   ✓ Removed old model for fresh training")
            
            success = run_training()
            if not success:
                print(f"   ✗ Skipping this experiment due to training failure")
                continue
        
        # Evaluation
        result = run_evaluation()
        if result is None:
            print(f"   ✗ Skipping this experiment due to evaluation failure")
            continue
        
        reward, std = result
        
        # Save and display
        exp_result = save_experiment_result(exp, reward, std, backup_file)
        results.append(exp_result)
        print_result(exp, reward, std)
        
        print(f"\n   ✓ Experiment {exp['id']} complete!")
    
    # Final summary
    if results:
        print_summary(results)
    else:
        print(f"\n   ✗ No experiments completed successfully")
    
    print_header("✅ OPTIMIZATION COMPLETE")
    
    print(f"\n📁 Results saved to:")
    print(f"   results/optimization_EXP-*.json  (individual results)")
    print(f"   results/optimization_summary.json  (complete summary)")
    
    print(f"\n📝 Next Steps:")
    print(f"   1. Review results in optimization_summary.json")
    print(f"   2. Select best configuration")
    print(f"   3. Document experiments in Chapter 3")
    print(f"   4. Use best model for final thesis evaluation")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Optimization interrupted by user")
        print(f"   Partial results may be in results/ folder")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
#!/usr/bin/env python3
"""
Quick Optimization Study - Using YOUR actual Actuated baseline
ACTUATED_BASELINE = 20940.65 (from your COMPREHENSIVE_EVALUATION results)
"""

import json
import numpy as np
from src.simulation.environment import JammingMachine
from src.agent.q_learning_agent import QLearningAgent

# ============================================
# CONFIGURATION
# ============================================

NUM_EVAL_EPISODES = 100
SIM_CONFIG_PATH = "config/sim_config.json"
Q_TABLE_PATH = "trained_q_table.json"

# ============================================
# ✅ YOUR ACTUATED BASELINE FROM NOTEBOOK
# ============================================

ACTUATED_BASELINE = 20940.65  # From your results table!

# ============================================
# 5 CONFIGURATIONS TO TEST
# ============================================

CONFIGS = [
    {
        'name': 'Config 1 - Baseline',
        'weights': {'throughput': 0.40, 'queue': 0.30, 'waiting': 0.20, 'fairness': 0.10}
    },
    {
        'name': 'Config 2 - Moderate Throughput',
        'weights': {'throughput': 0.50, 'queue': 0.25, 'waiting': 0.20, 'fairness': 0.05}
    },
    {
        'name': 'Config 3 - High Throughput',
        'weights': {'throughput': 0.55, 'queue': 0.25, 'waiting': 0.15, 'fairness': 0.05}
    },
    {
        'name': 'Config 4 - Very High Throughput',
        'weights': {'throughput': 0.60, 'queue': 0.20, 'waiting': 0.15, 'fairness': 0.05}
    },
    {
        'name': 'Config 5 - Extreme Throughput',
        'weights': {'throughput': 0.70, 'queue': 0.15, 'waiting': 0.10, 'fairness': 0.05}
    }
]

# ============================================
# HELPER FUNCTIONS
# ============================================

def load_trained_agent(q_table_path, state_space_size=8, action_space_size=6):
    """Load pre-trained Q-Learning agent"""
    print(f"\n📂 Loading trained Q-table from: {q_table_path}")
    
    with open(q_table_path, 'r') as f:
        q_table_data = json.load(f)
    
    agent = QLearningAgent(
        state_space_size=state_space_size,
        action_space_size=action_space_size,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=0.0  # No exploration during eval
    )
    
    # Convert string keys back to tuples
    agent.q_table = {
        tuple(map(int, k.strip('()').split(','))): v 
        for k, v in q_table_data.items()
    }
    
    print(f"✓ Loaded {len(agent.q_table)} state-action pairs\n")
    return agent


def evaluate_with_config(agent, config_weights, num_episodes):
    """Evaluate agent with specific reward configuration"""
    
    # Load base config
    with open(SIM_CONFIG_PATH, 'r') as f:
        sim_config = json.load(f)
    
    # Update reward weights
    sim_config['reward_weights'] = config_weights
    
    # Create environment
    env = JammingMachine(sim_config)
    
    # Evaluate
    rewards = []
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Handle missing states with default action
            try:
                action = agent.choose_action(state)
            except KeyError:
                # If state not in Q-table, use default action (NS Normal)
                action = 1  # Default to NS Normal (30s green)
            
            state, reward, done, info = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
        
        if (ep + 1) % 25 == 0:
            print(f"    Episode {ep+1}/{num_episodes}: "
                  f"Avg = {np.mean(rewards):.2f}")
    
    return {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'rewards': rewards
    }


# ============================================
# MAIN EVALUATION
# ============================================

def main():
    print("="*80)
    print(" 🎯 QUICK OPTIMIZATION STUDY")
    print(" Evaluating existing Q-table with different reward configurations")
    print("="*80)
    print(f"\n Actuated Baseline: {ACTUATED_BASELINE:.2f}")
    print(f" (Your Q-Learning current result: 20,879.22 = -0.3% gap)")
    print("="*80)
    
    # Load trained agent once
    agent = load_trained_agent(Q_TABLE_PATH)
    
    # Store results
    results = []
    
    # Evaluate each configuration
    for i, config in enumerate(CONFIGS, 1):
        print(f"\n{'='*80}")
        print(f"📊 Config {i}/5: {config['name']}")
        print(f"   Weights: T={config['weights']['throughput']:.2f}, "
              f"Q={config['weights']['queue']:.2f}, "
              f"W={config['weights']['waiting']:.2f}, "
              f"F={config['weights']['fairness']:.2f}")
        print(f"{'='*80}\n")
        
        eval_results = evaluate_with_config(
            agent, 
            config['weights'], 
            NUM_EVAL_EPISODES
        )
        
        gap = ACTUATED_BASELINE - eval_results['mean']
        gap_pct = (gap / ACTUATED_BASELINE) * 100
        
        result = {
            'config_num': i,
            'name': config['name'],
            'weights': config['weights'],
            'mean_reward': eval_results['mean'],
            'std_reward': eval_results['std'],
            'gap': gap,
            'gap_percent': gap_pct
        }
        
        results.append(result)
        
        print(f"\n  ✓ Results:")
        print(f"    Mean Reward:  {eval_results['mean']:.2f} ± {eval_results['std']:.2f}")
        print(f"    vs Actuated:  Gap = {gap:.2f} points ({gap_pct:+.2f}%)")
    
    # ============================================
    # SUMMARY TABLE
    # ============================================
    
    print(f"\n\n{'='*80}")
    print(" 📈 OPTIMIZATION STUDY RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Config':<8} {'w_T':<6} {'Mean Reward':<16} {'Gap':<12} {'Improvement':<12}")
    print("-" * 80)
    
    baseline_gap = results[0]['gap']
    
    for r in results:
        improvement = ((baseline_gap - r['gap']) / abs(baseline_gap)) * 100 if baseline_gap != 0 else 0
        marker = "⭐" if r['config_num'] == 1 else ""
        
        print(f"{r['config_num']:<8} "
              f"{r['weights']['throughput']:<6.2f} "
              f"{r['mean_reward']:<16.2f} "
              f"{r['gap']:<+12.2f} "
              f"{improvement:<+12.1f}% {marker}")
    
    # ============================================
    # LINEAR REGRESSION ANALYSIS
    # ============================================
    
    print(f"\n{'='*80}")
    print(" 🔬 LINEAR RELATIONSHIP ANALYSIS")
    print(f"{'='*80}\n")
    
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    X = np.array([r['weights']['throughput'] for r in results]).reshape(-1, 1)
    y = np.array([r['gap'] for r in results])
    
    model = LinearRegression()
    model.fit(X, y)
    r_squared = r2_score(y, model.predict(X))
    
    print(f"  Linear Equation: Gap = {model.coef_[0]:.2f} × w_T + {model.intercept_:.2f}")
    print(f"  R² Score: {r_squared:.4f}")
    print(f"  ")
    print(f"  Interpretation:")
    print(f"    • Each 0.05 increase in throughput weight")
    print(f"      → {model.coef_[0]*0.05:+.2f} point gap change")
    
    if r_squared > 0.95:
        print(f"    • Strong linear relationship (predictable)")
    elif r_squared > 0.90:
        print(f"    • Moderate linear relationship")
    else:
        print(f"    • Weak relationship (but still informative)")
    
    # ============================================
    # SAVE RESULTS
    # ============================================
    
    output = {
        'actuated_baseline': ACTUATED_BASELINE,
        'baseline_ql_result': 20879.22,
        'baseline_gap': 61.43,
        'baseline_gap_percent': -0.3,
        'num_episodes': NUM_EVAL_EPISODES,
        'configs': results,
        'linear_model': {
            'slope': float(model.coef_[0]),
            'intercept': float(model.intercept_),
            'r_squared': float(r_squared)
        }
    }
    
    with open('optimization_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f" ✅ COMPLETE!")
    print(f"{'='*80}")
    print(f"\n  Results saved to: optimization_results.json")
    print(f"  Next step: Create Figure 4 and write Section 4.2\n")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
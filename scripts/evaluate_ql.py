#!/usr/bin/env python3
"""Q-Learning Agent Evaluation"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


import json
import numpy as np
from src.simulation.environment import JammingMachine
from src.agent.q_learning_agent import QLearningAgent

def run_evaluation(episodes=500):
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
    
    print(f"\nQ-Learning Results: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    return summary

if __name__ == "__main__":
    run_evaluation()

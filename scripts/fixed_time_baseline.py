#!/usr/bin/env python3
"""Fixed-Time Traffic Controller Baseline"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


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

def run_evaluation(episodes=500):
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
    
    print(f"\nFixed-Time Results: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    return summary

if __name__ == "__main__":
    run_evaluation()

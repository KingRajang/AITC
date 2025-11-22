#!/usr/bin/env python3
"""Actuated Traffic Controller Baseline"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


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

def run_evaluation(episodes=500):
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
    
    print(f"\nActuated Results: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    return summary

if __name__ == "__main__":
    run_evaluation()

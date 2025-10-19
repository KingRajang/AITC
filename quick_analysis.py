#!/usr/bin/env python3
"""
Quick Analysis Script for AITC Training Results
Displays key metrics without needing Jupyter
"""

import json
import numpy as np

def main():
    print("=" * 60)
    print("AITC Project - Quick Training Analysis")
    print("=" * 60)
    
    # Load training log
    try:
        with open('training_log.json', 'r') as f:
            training_log = json.load(f)
        
        print(f"\n📊 Training Overview:")
        print(f"  • Total Episodes: {len(training_log)}")
        print(f"  • Training Status: {'Completed' if len(training_log) >= 10000 else 'In Progress'}")
        
        # Extract rewards
        rewards = [episode['total_reward'] for episode in training_log]
        
        print(f"\n🎯 Reward Statistics:")
        print(f"  • First Episode Reward: {rewards[0]:,.2f}")
        print(f"  • Final Episode Reward: {rewards[-1]:,.2f}")
        print(f"  • Average Reward: {np.mean(rewards):,.2f}")
        print(f"  • Best Reward: {np.max(rewards):,.2f}")
        print(f"  • Worst Reward: {np.min(rewards):,.2f}")
        
        # Learning trend (compare first 1000 vs last 1000)
        if len(rewards) >= 2000:
            early_avg = np.mean(rewards[:1000])
            late_avg = np.mean(rewards[-1000:])
            improvement = ((late_avg - early_avg) / abs(early_avg)) * 100
            
            print(f"\n📈 Learning Progress:")
            print(f"  • Early Episodes (1-1000): {early_avg:,.2f}")
            print(f"  • Late Episodes ({len(rewards)-999}-{len(rewards)}): {late_avg:,.2f}")
            print(f"  • Performance Change: {improvement:+.2f}%")
        
        # Exploration decay
        print(f"\n🔍 Exploration:")
        print(f"  • Initial Epsilon: {training_log[0]['epsilon']:.3f}")
        print(f"  • Final Epsilon: {training_log[-1]['epsilon']:.3f}")
        
    except FileNotFoundError:
        print("❌ training_log.json not found. Please run training first.")
        return
    
    # Load initial state
    try:
        with open('data/initial_state.json', 'r') as f:
            initial_state = json.load(f)
        
        print(f"\n🚦 Initial Traffic State (from Vision Analysis):")
        for lane, state in initial_state.items():
            density_desc = "Congested" if state['density_score'] >= 0.8 else "Smooth" if state['density_score'] <= 0.4 else "Medium"
            print(f"  • {lane}: {state['vehicle_count']} vehicles, {density_desc} ({state['density_score']})")
    
    except FileNotFoundError:
        print("⚠️  Initial state not found. Run vision processing first.")
    
    print(f"\n📁 Generated Files:")
    files_generated = [
        ("trained_q_table.json", "Trained Q-Learning policy"),
        ("training_log.json", "Episode-by-episode training log"),
        ("data/initial_state.json", "Traffic state from vision analysis"),
        ("data/output_images/", "Annotated traffic images")
    ]
    
    for filename, description in files_generated:
        try:
            import os
            if os.path.exists(filename):
                print(f"  ✅ {filename} - {description}")
            else:
                print(f"  ❌ {filename} - {description}")
        except:
            print(f"  ? {filename} - {description}")
    
    print(f"\n🎓 Next Steps:")
    print(f"  1. Open Jupyter Lab: jupyter lab")
    print(f"  2. Run notebooks/2_results_analysis.ipynb")
    print(f"  3. View annotated images in data/output_images/")
    print(f"  4. Compare with baseline fixed-time controller")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

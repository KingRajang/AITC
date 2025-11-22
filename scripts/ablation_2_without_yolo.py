import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.simulation.environment import JammingMachine
from src.agent.q_learning_agent import QLearningAgent

print("="*70)
print("ABLATION 2: WITHOUT YOLO")
print("="*70)

with open('config/sim_config.json') as f: sim_config = json.load(f)
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

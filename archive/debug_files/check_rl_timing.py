import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import sys
import json
import numpy as np
from src.simulation.environment import JammingMachine
from src.agent.q_learning_agent import QLearningAgent

print("=" * 70)
print("ANALYZING RL AGENT BEHAVIOR")
print("=" * 70)

with open('config/sim_config.json') as f:
    config = json.load(f)

env = JammingMachine(config)
agent = QLearningAgent(8, 6, 0, 0.95, 0)
agent.load_q_table('trained_q_table.json')

# Run one episode and track action patterns
state = env.reset()
done = False
actions = []
action_durations = []
current_action = None
current_duration = 0

step = 0
while not done and step < 300:
    action = agent.choose_action(state)
    actions.append(action)
    
    # Track action switches
    if action != current_action:
        if current_action is not None:
            action_durations.append(current_duration)
        current_action = action
        current_duration = 1
    else:
        current_duration += 1
    
    state, reward, done, info = env.step(action)
    step += 1

# Add last duration
if current_duration > 0:
    action_durations.append(current_duration)

print("\nACTION STATISTICS:")
print(f"Total steps: {len(actions)}")
print(f"Number of action switches: {len(action_durations)}")
print(f"Average duration per action: {np.mean(action_durations):.1f} steps")
print(f"Min duration: {min(action_durations)}")
print(f"Max duration: {max(action_durations)}")
print(f"Median duration: {np.median(action_durations):.1f}")

print("\nACTION DISTRIBUTION:")
action_counts = {}
for a in actions:
    action_counts[a] = action_counts.get(a, 0) + 1

for action, count in sorted(action_counts.items()):
    pct = (count / len(actions)) * 100
    print(f"Action {action}: {count} times ({pct:.1f}%)")

print("\n" + "=" * 70)
print("COMPARISON TO BASELINES:")
print("=" * 70)

avg_duration = np.mean(action_durations)
print(f"\nRL avg action duration: {avg_duration:.1f} steps")
print(f"15s baseline duration: 15 steps (BEST: 59.67/step)")
print(f"30s baseline duration: 30 steps (48.75/step)")
print(f"45s baseline duration: 45 steps (WORST: 45.32/step)")

if avg_duration < 20:
    print("\n✅ RL switches relatively fast (like 15s baseline)")
    print("   But still performs worse - other issues at play")
elif avg_duration < 35:
    print("\n⚠️  RL switches at moderate rate (like 30s baseline)")
    print("   This explains why performance similar to 30s")
elif avg_duration >= 35:
    print("\n❌ RL switches very slowly (like 45s+ baseline)")
    print("   This explains poor performance!")

print("=" * 70)

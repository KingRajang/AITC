import json
import numpy as np
from collections import Counter

with open('trained_q_table.json') as f:
    qtable = json.load(f)

print("=" * 70)
print("Q-TABLE ACTION PREFERENCE ANALYSIS")
print("=" * 70)

best_actions = []
action_0_is_best = 0
all_same = 0

for state, q_vals in qtable.items():
    best_action = np.argmax(q_vals)
    best_actions.append(best_action)
    
    if best_action == 0:
        action_0_is_best += 1
    
    # Check if all Q-values are same (within tolerance)
    if np.std(q_vals) < 1:
        all_same += 1

print(f"\nTotal states: {len(qtable)}")
print(f"States where Action 0 is best: {action_0_is_best} ({action_0_is_best/len(qtable)*100:.1f}%)")
print(f"States with nearly identical Q-values: {all_same} ({all_same/len(qtable)*100:.1f}%)")

print("\nBest action distribution:")
action_dist = Counter(best_actions)
for action in range(6):
    count = action_dist.get(action, 0)
    pct = (count / len(best_actions)) * 100
    bar = "█" * int(pct / 2)
    print(f"Action {action}: {count:3d} ({pct:5.1f}%) {bar}")

print("\n" + "=" * 70)

if action_0_is_best > len(qtable) * 0.8:
    print("❌ CRITICAL: 80%+ of states prefer Action 0!")
    print("   Training converged to Action 0 bias")
    print("   This is why evaluation always picks Action 0")
elif action_0_is_best > len(qtable) * 0.5:
    print("⚠️  WARNING: 50%+ of states prefer Action 0")
    print("   Moderate Action 0 bias exists")
else:
    print("✅ Action preferences are relatively balanced")
    print("   Problem must be elsewhere")

print("=" * 70)

# Sample some states with their best actions
print("\nSAMPLE STATES AND BEST ACTIONS:")
import random
samples = random.sample(list(qtable.items()), min(10, len(qtable)))
for state, q_vals in samples:
    best = np.argmax(q_vals)
    print(f"\nState: {state}")
    print(f"Q-values: {[f'{v:.1f}' for v in q_vals]}")
    print(f"Best: Action {best} (Q={q_vals[best]:.1f})")

print("=" * 70)

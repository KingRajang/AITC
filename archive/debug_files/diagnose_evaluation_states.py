import sys
sys.path.append('src')
import json
from simulation.environment import JammingMachine
from agent.q_learning_agent import QLearningAgent

print("=" * 70)
print("EVALUATION STATE COVERAGE DIAGNOSIS")
print("=" * 70)

with open('config/sim_config.json') as f:
    config = json.load(f)

env = JammingMachine(config)
agent = QLearningAgent(8, 6, 0, 0.95, 0)
agent.load_q_table('trained_q_table.json')

print(f"\n✅ Q-table loaded: {len(agent.q_table)} states")
print(f"✅ Epsilon set to: {agent.epsilon}")

# Run evaluation and track states
state = env.reset()
states_encountered = []
states_in_qtable = []
actions_taken = []
q_value_sources = []

for step in range(50):  # First 50 steps
    # Discretize current state
    discrete_state = agent._discretize_state(state)
    states_encountered.append(discrete_state)
    
    # Check if in Q-table
    in_qtable = discrete_state in agent.q_table
    states_in_qtable.append(in_qtable)
    
    # Get Q-values
    if in_qtable:
        q_values = agent.q_table[discrete_state]
        source = "Q-TABLE"
    else:
        q_values = [0] * agent.action_space_size
        source = "DEFAULT"
    
    q_value_sources.append(source)
    
    # Choose action
    action = agent.choose_action(state)
    actions_taken.append(action)
    
    # Take step
    state, reward, done, info = env.step(action)
    if done:
        break

# Analysis
print(f"\n📊 EVALUATION RESULTS (first {len(states_encountered)} steps):")
print(f"States encountered: {len(states_encountered)}")
print(f"States in Q-table: {sum(states_in_qtable)} ({sum(states_in_qtable)/len(states_in_qtable)*100:.1f}%)")
print(f"States NOT in Q-table: {len(states_in_qtable) - sum(states_in_qtable)} ({(1-sum(states_in_qtable)/len(states_in_qtable))*100:.1f}%)")

print(f"\n🎯 ACTION DISTRIBUTION:")
from collections import Counter
action_dist = Counter(actions_taken)
for action in range(6):
    count = action_dist.get(action, 0)
    print(f"Action {action}: {count} times")

print(f"\n🔍 DETAILED STATE ANALYSIS (first 10 steps):")
print(f"{'Step':<6} {'In Q-table?':<12} {'Source':<12} {'Action':<8} {'State (first 4 dims)':<30}")
print("-" * 80)

for i in range(min(10, len(states_encountered))):
    state_str = str(states_encountered[i][:4]) + "..."
    in_q = "✅ YES" if states_in_qtable[i] else "❌ NO"
    print(f"{i+1:<6} {in_q:<12} {q_value_sources[i]:<12} {actions_taken[i]:<8} {state_str:<30}")

# Find problem
print("\n" + "=" * 70)
print("DIAGNOSIS:")
print("=" * 70)

coverage = sum(states_in_qtable) / len(states_in_qtable) * 100
action_0_pct = (action_dist.get(0, 0) / len(actions_taken)) * 100

if coverage < 50:
    print("❌ CRITICAL: Less than 50% of evaluation states in Q-table!")
    print("   Agent hitting mostly unseen states → defaulting to Action 0")
    print("\n   ROOT CAUSE: State coverage mismatch")
    print("   SOLUTION: Need to improve state space coverage or generalization")
    
elif action_0_pct > 90 and coverage > 50:
    print("❌ CRITICAL: Most states in Q-table but still picking Action 0!")
    print("   Q-values might all favor Action 0")
    print("\n   ROOT CAUSE: Training bias toward Action 0")
    print("   SOLUTION: Check reward function and training process")
    
elif agent.epsilon > 0:
    print("❌ CRITICAL: Epsilon not set to 0 during evaluation!")
    print(f"   Current epsilon: {agent.epsilon}")
    print("\n   ROOT CAUSE: Still exploring during evaluation")
    print("   SOLUTION: Set epsilon=0 in evaluation code")
    
else:
    print("⚠️  UNKNOWN ISSUE")
    print(f"   Coverage: {coverage:.1f}%")
    print(f"   Action 0: {action_0_pct:.1f}%")
    print(f"   Epsilon: {agent.epsilon}")

# Check specific Q-values for encountered states
print("\n📋 SAMPLE Q-VALUES FOR ENCOUNTERED STATES:")
sample_count = 0
for i, (state, in_q) in enumerate(zip(states_encountered[:10], states_in_qtable[:10])):
    if in_q and sample_count < 3:
        q_vals = agent.q_table[state]
        best = max(range(len(q_vals)), key=lambda x: q_vals[x])
        print(f"\nStep {i+1}: {state}")
        print(f"  Q-values: {[f'{v:.1f}' for v in q_vals]}")
        print(f"  Best action: {best} (Q={q_vals[best]:.1f})")
        sample_count += 1

print("=" * 70)

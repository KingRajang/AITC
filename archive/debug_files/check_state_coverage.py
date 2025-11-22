import json
import sys
sys.path.append('src')

from simulation.environment import JammingMachine
from agent.q_learning_agent import QLearningAgent

# Load
with open('config/sim_config.json') as f:
    config = json.load(f)

env = JammingMachine(config)
agent = QLearningAgent(8, 6, 0, 0.95, 0)
agent.load_q_table('trained_q_table.json')

print("="*60)
print("STATE COVERAGE CHECK")
print("="*60)

# Run one episode and track states
state = env.reset()
states_encountered = []
states_in_qtable = []

for step in range(50):  # First 50 steps
    states_encountered.append(state)
    
    # Check if state in Q-table
    state_key = str(state) if isinstance(state, tuple) else tuple(state)
    if str(state_key) in agent.q_table:
        states_in_qtable.append(True)
    else:
        states_in_qtable.append(False)
    
    action = agent.choose_action(state)
    state, reward, done, info = env.step(action)
    
    if done:
        break

coverage = sum(states_in_qtable) / len(states_in_qtable) * 100

print(f"\nStates encountered: {len(states_encountered)}")
print(f"States in Q-table: {sum(states_in_qtable)}")
print(f"Coverage: {coverage:.1f}%")

if coverage < 50:
    print("\n❌ PROBLEM: Most states NOT in Q-table!")
    print("   Agent is making decisions for unseen states")
    print("   This explains poor performance")
elif coverage < 80:
    print("\n⚠️  WARNING: Many states not in Q-table")
    print("   Agent struggling with some situations")
else:
    print("\n✅ GOOD: Most states are in Q-table")

# Show sample of encountered vs known states
print("\n" + "="*60)
print("SAMPLE STATES (first 10 steps)")
print("="*60)
for i, (state, in_table) in enumerate(zip(states_encountered[:10], states_in_qtable[:10])):
    status = "✓" if in_table else "✗"
    print(f"Step {i+1}: {status} {state}")

print("="*60)

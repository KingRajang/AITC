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
print("STATE COVERAGE CHECK (FIXED)")
print("="*60)

# Run one episode and track states
state = env.reset()
states_encountered = []
discrete_states = []
states_in_qtable = []

for step in range(50):
    # Store raw state
    states_encountered.append(state.copy())
    
    # DISCRETIZE the state (this is what agent actually uses!)
    discrete_state = agent._discretize_state(state)
    discrete_states.append(discrete_state)
    
    # Check if DISCRETIZED state in Q-table
    if discrete_state in agent.q_table:
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
elif coverage < 80:
    print("\n⚠️  WARNING: Many states not in Q-table")
    print("   Agent struggling with some situations")
else:
    print("\n✅ GOOD: Most states are in Q-table")

# Show sample
print("\n" + "="*60)
print("SAMPLE (first 10 steps)")
print("="*60)
print("Step | Raw State | Discrete State | In Q-table?")
print("-" * 60)
for i in range(min(10, len(states_encountered))):
    raw = states_encountered[i]
    disc = discrete_states[i]
    in_table = "✓" if states_in_qtable[i] else "✗"
    print(f"{i+1:4d} | {str(raw)[:30]:30s} | {str(disc)[:20]:20s} | {in_table}")

print("="*60)

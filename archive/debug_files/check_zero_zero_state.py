import json

with open('trained_q_table.json') as f:
    qtable = json.load(f)

print("=" * 70)
print("CHECKING SPECIFIC EVALUATION STATES")
print("=" * 70)

# States that appeared during evaluation
eval_states = [
    "(0, 0, 0, 0, 0, 0, 0, 0)",
    "(0, 0, 0, 0, 2, 0, 2, 0)",
    "(0, 0, 0, 0, 3, 0, 3, 0)",
]

print("\nChecking if evaluation states are in Q-table:")
for state_str in eval_states:
    if state_str in qtable:
        q_vals = qtable[state_str]
        print(f"\n✅ {state_str} IS in Q-table")
        print(f"   Q-values: {q_vals}")
        
        if all(v == 0.0 for v in q_vals):
            print(f"   ❌ All zeros! This is the problem!")
        elif all(v == 100.0 for v in q_vals):
            print(f"   ⚠️  All 100s (never updated)")
        else:
            print(f"   ✅ Properly trained")
    else:
        print(f"\n❌ {state_str} NOT in Q-table")
        print(f"   Would get default values")

# Count zero-value states
zero_states = 0
hundred_states = 0
trained_states = 0

for state, q_vals in qtable.items():
    if all(v == 0.0 for v in q_vals):
        zero_states += 1
    elif all(abs(v - 100.0) < 1.0 for v in q_vals):
        hundred_states += 1
    else:
        trained_states += 1

print("\n" + "=" * 70)
print("Q-TABLE STATE BREAKDOWN:")
print("=" * 70)
print(f"Total states: {len(qtable)}")
print(f"Zero-value states (0,0,0,0,0,0): {zero_states} ({zero_states/len(qtable)*100:.1f}%)")
print(f"Hundred-value states (100,100,...): {hundred_states} ({hundred_states/len(qtable)*100:.1f}%)")  
print(f"Trained states (updated values): {trained_states} ({trained_states/len(qtable)*100:.1f}%)")

if zero_states > 0:
    print(f"\n❌ PROBLEM CONFIRMED!")
    print(f"   {zero_states} states still have zero values from OLD Q-table")
    print(f"   These are being loaded from file, not created fresh")
    print(f"\n   SOLUTION: Delete old Q-table and retrain from scratch!")

print("=" * 70)

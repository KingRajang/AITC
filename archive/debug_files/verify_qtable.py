import json

with open('trained_q_table.json') as f:
    q_table = json.load(f)

print("="*60)
print("Q-TABLE VERIFICATION")
print("="*60)

print(f"\nQ-table size: {len(q_table)} states")

# Check sample Q-values
sample_state = list(q_table.keys())[0]
q_values = q_table[sample_state]

print(f"\nSample state: {sample_state}")
print(f"Q-values: {[f'{v:.2f}' for v in q_values]}")
print(f"Max Q-value: {max(q_values):.2f}")

# Check if values are in new scale
avg_q = sum([max(q_table[s]) for s in list(q_table.keys())[:50]]) / 50

print(f"\nAverage max Q-value (first 50 states): {avg_q:.2f}")

if 500 < avg_q < 2000:
    print("✅ NEW Q-TABLE! (Values in correct range for new reward scale)")
elif 10000 < avg_q < 30000:
    print("❌ OLD Q-TABLE! (Values still in old scale)")
else:
    print(f"⚠️  UNEXPECTED scale: {avg_q}")

print("="*60)

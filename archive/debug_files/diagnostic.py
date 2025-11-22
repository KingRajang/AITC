import json
import pandas as pd
import matplotlib.pyplot as plt

print("="*60)
print("TRAINING DIAGNOSTIC")
print("="*60)

# Load training log
with open('training_log.json') as f:
    log = json.load(f)

df = pd.DataFrame(log)

print(f"\nTotal Episodes: {len(df)}")
print(f"Initial Reward (first 100): {df['total_reward'].iloc[:100].mean():.2f}")
print(f"Final Reward (last 100): {df['total_reward'].iloc[-100:].mean():.2f}")
improvement = df['total_reward'].iloc[-100:].mean() - df['total_reward'].iloc[:100].mean()
print(f"Improvement: {improvement:+.2f}")
print(f"Final Epsilon: {df['epsilon'].iloc[-1]:.6f}")

if improvement > 0:
    print("\n✅ Training improved")
else:
    print("\n❌ Training did NOT improve")

# Check Q-table
with open('trained_q_table.json') as f:
    q_table = json.load(f)

print(f"\nQ-Table Size: {len(q_table)} states")

if len(q_table) > 0:
    sample_state = list(q_table.keys())[0]
    q_values = q_table[sample_state]
    print(f"Sample Q-values: {[f'{v:.3f}' for v in q_values]}")
    
    if all(v == 0 for v in q_values):
        print("❌ Q-VALUES ARE ALL ZEROS")
    else:
        print("✅ Q-values learned")

print("="*60)

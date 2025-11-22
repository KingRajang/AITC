import json, numpy as np
from scipy import stats

configs = {'Full System': 'results/ablation_1_full_system.json',
           'Without YOLO': 'results/ablation_2_without_yolo.json',
           'Without DBSCAN': 'results/ablation_3_without_dbscan.json',
           'Without RL': 'results/ablation_4_without_rl.json'}

results = {}
for name, path in configs.items():
    with open(path) as f: results[name] = json.load(f)

print("\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)
for name in configs.keys():
    r = results[name]
    print(f"{name:<20} {r['mean_reward']:>8.2f} ± {r['std_reward']:<6.2f}")

baseline = results['Full System']['mean_reward']
print("\n" + "="*80)
print("COMPONENT CONTRIBUTIONS")
print("="*80)
for comp, key in [('YOLO', 'Without YOLO'), ('DBSCAN', 'Without DBSCAN'), ('RL', 'Without RL')]:
    deg = ((baseline - results[key]['mean_reward']) / baseline) * 100
    print(f"{comp:<10} {deg:>6.2f}% {'🔴 CRITICAL' if deg > 20 else '⚠️ IMPORTANT' if deg > 5 else '✓'}")

print("\n" + "="*80)
print("STATISTICAL TESTS (vs Full System)")
print("="*80)
full_r = np.array(results['Full System']['detailed_results']['total_rewards'])
for name in ['Without YOLO', 'Without DBSCAN', 'Without RL']:
    config_r = np.array(results[name]['detailed_results']['total_rewards'])
    t, p = stats.ttest_ind(full_r, config_r)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"{name:<20} t={t:>7.2f}  p={p:.6f} {sig}")
print("\n✅ Ablation Analysis Complete!")

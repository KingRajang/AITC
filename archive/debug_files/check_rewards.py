import json

# Load sim_config to see reward configuration
with open('config/sim_config.json') as f:
    config = json.load(f)

print("="*60)
print("REWARD CONFIGURATION CHECK")
print("="*60)

if 'reward_weights' in config:
    print("\n✅ Using weighted reward system:")
    for component, weight in config['reward_weights'].items():
        print(f"  {component}: {weight}")
else:
    print("\n⚠️  No reward_weights found - using simple reward")

if 'reward_calculator' in config:
    print(f"\nReward calculator: {config['reward_calculator']}")
    
print("\n" + "="*60)

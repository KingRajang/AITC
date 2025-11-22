# fix_imports.py
import os

scripts = [
    'scripts/evaluate_ql.py',
    'scripts/actuated_baseline.py', 
    'scripts/fixed_time_baseline.py'
]

fix = """import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

"""

for script in scripts:
    with open(script, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if 'sys.path.append' not in content:
        lines = content.split('\n')
        # Insert after shebang and docstring
        insert_pos = 2 if lines[0].startswith('#!') else 1
        lines.insert(insert_pos, fix)
        
        with open(script, 'w') as f:
            f.write('\n'.join(lines))
        print(f"✓ Fixed: {script}")
    else:
        print(f"✓ Already fixed: {script}")

print("\nNow run: python run_experiments.py")
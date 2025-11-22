# QUICK START GUIDE

## 1️⃣ Setup (Already Done!)
You just ran setup_experiments.py ✓

## 2️⃣ Run Experiments

### Option A: All at once (Recommended)
```bash
python run_experiments.py
```
This will:
- Train Q-Learning (if not done)
- Evaluate all 3 controllers
- Generate comparison statistics
- Create comparison plot

**Time:** 4-5 hours total

### Option B: Step by step
```bash
# Train Q-Learning
python scripts/main_rl_training.py

# Evaluate all controllers
python scripts/evaluate_ql.py
python scripts/actuated_baseline.py
python scripts/fixed_time_baseline.py

# Compare results
python scripts/compare_results.py
```

## 3️⃣ Check Results
```bash
# View summary
cat results/q_learning_evaluation.json
cat results/actuated_baseline.json
cat results/fixed_time_baseline.json

# View plot
open results/comparison_plot.png
```

## 4️⃣ For Paper
Results are in `results/` folder:
- Statistical comparison printed to console
- Plot ready for paper in `results/comparison_plot.png`
- Raw data in JSON files

## 🚨 Troubleshooting

**ImportError?**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Files not found?**
Make sure you're in the project root directory

**Still stuck?**
Run step by step (Option B) to identify the issue

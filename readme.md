# AITC: AI-Powered Adaptive Traffic Control

Welcome to the **AITC project repository!** This framework is the official implementation for the thesis project:  
**"Traffic Control Using Artificial Intelligence For Adaptive And Optimized Traffic Management."**

## 📋 Table of Contents

- [Project Overview & Goal](#1-project-overview--goal)
- [System Architecture & Logic](#2-system-architecture--logic)
- [Project Structure](#3-project-directory-structure)
- [Installation & Setup](#4-installation--setup)
- [Running the Experiments](#5-running-the-experiments)
- [Configuration Files](#6-configuration-files)
- [Analysis & Visualization](#7-analysis--visualization)
- [Experiment Management](#8-experiment-management)
- [Current Status & Next Steps](#9-current-status--next-steps)
- [Troubleshooting](#10-troubleshooting)

---

## 1. Project Overview & Goal

The goal of this project is to design, train, and evaluate an intelligent system that can adaptively control traffic signals in real-time. We achieve this by integrating **three distinct AI algorithms**, each handling a specific part of the problem:

1. **YOLOv8** - Real-time vehicle detection from traffic camera images
2. **DBSCAN** - Spatial clustering analysis for traffic density assessment
3. **Q-Learning** - Reinforcement learning agent for adaptive signal control

This modular design allows us to test and improve each part of the system independently.

---

## 2. System Architecture & Logic

Our system is built on a **three-phase methodology** that creates a pipeline from visual data to intelligent action.

### 🔍 Phase 1: Vision Pipeline (State Assessment)

**Purpose:** Analyze real-world traffic conditions from images

**Process:**

1. Load traffic images from each lane (North, South, East, West)
2. **YOLOv8** (`src/vision/yolo_processor.py`) detects vehicles in images
3. **DBSCAN** (`src/vision/dbscan_analyzer.py`) clusters vehicle locations and calculates:
   - `vehicle_count` - Total number of vehicles detected
   - `density_score` - Traffic density metric (0.0 to 1.0)

**Output:** `data/initial_state.json` - Initial traffic state for all lanes

---

### 🛠️ Phase 2: "Jamming Machine" (Simulation Environment)

**Purpose:** Fast, safe training environment that avoids expensive real-time vision processing

**Process:**

1. `JammingMachine` (`src/simulation/environment.py`) creates simulation environment
2. Loads initial traffic state from Phase 1
3. Uses stochastic traffic models (`src/simulation/traffic_models.py`) to simulate:
   - Vehicle arrivals (Poisson-like process)
   - Vehicle departures (based on signal states)
4. Computes reward signals using `RewardCalculator` (`src/simulation/reward_calculator.py`)

**Features:**

- Configurable traffic patterns (light/medium/heavy)
- Weighted multi-objective reward function
- Realistic saturation flow rates

---

### 🧠 Phase 3: Q-Learning Agent (Policy Optimization)

**Purpose:** Learn optimal traffic light control strategy

**Process:**

1. `QLearningAgent` (`src/agent/q_learning_agent.py`) interacts with simulator
2. For each episode:
   - Observes current state (vehicle counts + density scores)
   - Chooses action using ε-greedy policy
   - Receives reward based on traffic flow performance
   - Updates Q-table using Bellman equation
3. Gradually reduces exploration (ε decay)

**Output:**

- `models/trained_q_table.json` - Learned optimal policy
- `training_log.json` - Training progress and metrics

---

## 3. Project Directory Structure

```plaintext
aitc/
│
├── 📁 src/                        # Core source code (properly packaged)
│   ├── __init__.py
│   │
│   ├── vision/                    # Phase 1: Computer Vision
│   │   ├── __init__.py
│   │   ├── yolo_processor.py      # YOLOv8 vehicle detection
│   │   └── dbscan_analyzer.py     # DBSCAN clustering analysis
│   │
│   ├── simulation/                # Phase 2: Traffic Simulation
│   │   ├── __init__.py
│   │   ├── environment.py         # Jamming Machine simulator
│   │   ├── traffic_models.py      # Stochastic traffic flow models
│   │   └── reward_calculator.py   # Multi-objective reward function
│   │
│   └── agent/                     # Phase 3: Reinforcement Learning
│       ├── __init__.py
│       └── q_learning_agent.py    # Q-Learning RL agent
│
├── 📁 scripts/                    # Executable scripts
│   ├── main_vision_processing.py  # Phase 1 entry point
│   ├── main_rl_training.py        # Phase 2 & 3 entry point
│   ├── fresh_baseline_eval.py     # Baseline evaluation
│   ├── final_comprehensive_test.py    # Integration tests
│   └── check_rl_timing.py         # Performance profiling
│
├── 📁 config/                     # Configuration files
│   ├── lane_config.json           # Lane definitions and image mappings
│   ├── sim_config.json            # Simulation parameters
│   └── lane_mapping.json          # Lane ID to direction mapping
│
├── 📁 data/                       # Data directory
│   ├── input_images/              # (create this) Place traffic images here
│   ├── output_images/             # (auto-created) Annotated images
│   └── initial_state.json         # Generated initial traffic state
│
├── 📁 models/                     # Trained models storage
│   └── trained_q_table.json       # Latest trained Q-table
│
├── 📁 notebooks/                  # Jupyter notebooks for analysis
│   ├── 1_vision_pipeline_testbed.ipynb        # Test vision pipeline
│   ├── 2_results_analysis.ipynb               # Analyze training results
│   └── 3_enhanced_baseline_comparison.ipynb   # Compare with baselines
│
├── 📁 results/                    # Experiment results
│   └── detailed_results_*.json    # Timestamped experiment results
│
├── 📁 backup/                     # Backup of previous experiments
│   ├── exp1_10k_10bins/           # Experiment 1 backup
│   ├── exp2_10k_5bins_uniform/    # Experiment 2 backup
│   └── old_training_*/            # Old training runs
│
├── 📁 archive/                    # Archived code
│   ├── debug_files/               # Old debugging scripts
│   └── old_experiments/           # Previous experiment attempts
│
├── 📄 Root level files
│   ├── trained_q_table.json       # Latest training result
│   ├── training_log.json          # Latest training log
│   ├── check_state_coverage_FIXED.py  # State space analysis utility
│   ├── reorganize_project.py      # Project reorganization script
│   ├── requirements.txt           # Python dependencies
│   └── readme.md                  # This file
│
└── 📁 venv/                       # Virtual environment (not in git)
```

**Key Features of This Structure:**

- ✅ **Proper Python package** with `src/` folder and `__init__.py` files
- ✅ **Separation of concerns**: code, scripts, config, data, results
- ✅ **Version control friendly**: backups and archives organized
- ✅ **Reproducible experiments**: timestamped results and backups

---

## 4. Installation & Setup

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd aitc
```

### Step 2: Create Virtual Environment

**Using venv (recommended):**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Or using conda:**

```bash
conda create -n aitc python=3.10
conda activate aitc
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies installed:**

- `ultralytics` - YOLOv8 implementation
- `scikit-learn` - DBSCAN clustering
- `opencv-python` - Image processing
- `numpy`, `pandas` - Data manipulation
- `matplotlib` - Visualization
- `scipy` - Statistical analysis
- `jupyter`, `jupyterlab` - Interactive notebooks

### Step 4: Verify Installation

```bash
python -c "import ultralytics; import sklearn; import cv2; import pandas; import matplotlib; print('✓ All imports successful!')"
```

### Step 5: Create Required Directories

```bash
# Create input images directory if it doesn't exist
mkdir -p data/input_images
mkdir -p data/output_images
```

---

## 5. Running the Experiments

### 🎯 Quick Start (Full Pipeline)

```bash
# 1. Run vision processing to generate initial state
python scripts/main_vision_processing.py

# 2. Train the RL agent
python scripts/main_rl_training.py

# 3. Analyze results in Jupyter
jupyter lab
# Open: notebooks/2_results_analysis.ipynb
```

---

### 📸 Phase 1: Vision Processing (Detailed)

**Prerequisites:**
Place traffic images in `data/input_images/`:

- `lane_north.jpg`
- `lane_south.jpg`
- `lane_east.jpg`
- `lane_west.jpg`

**Run:**

```bash
python scripts/main_vision_processing.py
```

**Expected Output:**

```
--- Phase 1: Initial State Assessment (Multi-Image) ---
Loaded configuration for 4 lanes.
YOLOv8 Processor initialized for vehicle detection.

Processing Lane: 'lane_north' using image 'lane_north.jpg'...
  Detected 12 vehicles.
  Analysis complete. State: {'vehicle_count': 12, 'density_score': 0.5}
  Saved annotated image to: data/output_images/annotated_lane_north.jpg

[... similar output for other lanes ...]

Global initial state saved to 'data/initial_state.json'
--- Phase 1 Complete ---
```

**Generated Files:**

- `data/initial_state.json` - Traffic state data
- `data/output_images/annotated_*.jpg` - Visualizations with bounding boxes

---

### 🤖 Phase 2 & 3: RL Training (Detailed)

**Run:**

```bash
python scripts/main_rl_training.py
```

**Configuration (in `config/sim_config.json`):**

- `simulation_episodes`: Number of training episodes (default: 10,000)
- `max_steps_per_episode`: Steps per episode (default: 200)
- `learning_rate`: Q-learning alpha (default: 0.1)
- `discount_factor`: Q-learning gamma (default: 0.95)

**Expected Output:**

```
--- Phase 2 & 3: RL Agent Training ---
Loaded initial state from Phase 1 analysis.
Scaling down initial state...
  lane_north: 12 → 4 cars
  [...]

Jamming Machine environment and Q-Learning agent initialized.
Starting training for 10000 episodes...

Episode 100/10000 | Total Reward: 8245.32 | Avg/Step: 41.23 | Avg Queue: 8.5 | Epsilon: 0.9512
Episode 200/10000 | Total Reward: 8456.78 | Avg/Step: 42.28 | Avg Queue: 7.8 | Epsilon: 0.9048
[...]

Training complete.
Q-table successfully saved to models/trained_q_table.json
Training log saved to training_log.json
--- Phase 2 & 3 Complete ---
```

**Generated Files:**

- `models/trained_q_table.json` - Learned Q-values (can be large ~1MB)
- `training_log.json` - Episode-by-episode metrics
- Also copied to root: `trained_q_table.json`, `training_log.json`

---

## 6. Configuration Files

### 📋 config/lane_config.json

Defines lane-to-image mappings for vision processing:

```json
{
  "lanes": [
    { "lane_id": "lane_north", "image_filename": "lane_north.jpg" },
    { "lane_id": "lane_south", "image_filename": "lane_south.jpg" },
    { "lane_id": "lane_east", "image_filename": "lane_east.jpg" },
    { "lane_id": "lane_west", "image_filename": "lane_west.jpg" }
  ]
}
```

### ⚙️ config/sim_config.json

Controls simulation and training parameters:

```json
{
  "simulation_episodes": 10000,
  "max_steps_per_episode": 200,
  "lane_capacity": 50,
  "saturation_flow_rate": 4,
  "traffic_arrival_rates": {
    "light": [0, 2],
    "medium": [1, 3],
    "heavy": [2, 5]
  },
  "reward_weights": {
    "throughput": 0.4,
    "queue": 0.3,
    "waiting": 0.2,
    "fairness": 0.1
  },
  "normalization_params": {
    "max_throughput": 20,
    "max_waiting": 100,
    "max_queue_std": 20
  }
}
```

**Key Parameters to Tune:**

- `learning_rate`: Higher = faster learning but less stable (try 0.05-0.2)
- `discount_factor`: Higher = more long-term focused (try 0.9-0.99)
- `reward_weights`: Adjust priorities (must sum to 1.0)
- `epsilon_decay_rate`: Controls exploration-exploitation tradeoff

### 🗺️ config/lane_mapping.json

Maps internal lane IDs to cardinal directions:

```json
{
  "lane_mapping": {
    "lane_north": "North",
    "lane_south": "South",
    "lane_east": "East",
    "lane_west": "West"
  }
}
```

---

## 7. Analysis & Visualization

### 📊 Using Jupyter Notebooks

**Start Jupyter Lab:**

```bash
jupyter lab
```

The notebooks will automatically open in your browser.

---

### Notebook 1: Vision Pipeline Testing

**File:** `notebooks/1_vision_pipeline_testbed.ipynb`

**Purpose:** Test and debug vehicle detection pipeline

**Contents:**

- Load and display traffic images
- Run YOLOv8 detection with visualization
- Analyze DBSCAN clustering results
- Validate detection accuracy
- Generate annotated images

**Use Cases:**

- Debugging vision processing issues
- Testing different YOLO confidence thresholds
- Adjusting DBSCAN parameters (eps, min_samples)
- Visualizing detection results

---

### Notebook 2: Results Analysis (MAIN)

**File:** `notebooks/2_results_analysis.ipynb`

**Purpose:** Main analysis of training results

**Key Analyses:**

1. **Learning Curve:** Plot total reward over episodes
2. **Convergence Analysis:** Check if agent is improving
3. **Baseline Comparison:** Compare with fixed-time controller
4. **State-Action Analysis:** Examine Q-table patterns
5. **Performance Metrics:** Throughput, queue lengths, fairness

**Key Metrics Tracked:**

- Average reward per episode
- Average queue length over time
- Throughput (vehicles departed)
- Fairness (queue balance across lanes)
- Epsilon decay curve

**Example Workflow:**

```python
# Load training log
import json
with open('../training_log.json') as f:
    log = json.load(f)

# Plot learning curve
import matplotlib.pyplot as plt
rewards = [e['total_reward'] for e in log]
plt.plot(rewards)
plt.title('Learning Curve')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
```

---

### Notebook 3: Enhanced Baseline Comparison

**File:** `notebooks/3_enhanced_baseline_comparison.ipynb`

**Purpose:** Rigorous comparison with multiple baseline strategies

**Baselines Implemented:**

1. **Fixed-Time Controller:** Equal green time for all directions (30s/30s)
2. **Actuated Controller:** Vehicle-responsive timing
3. **Webster's Method:** Optimal timing based on traffic flow theory
4. **Random Policy:** Random action selection (sanity check)

**Comparison Metrics:**

- Average total waiting time
- Average queue length
- Throughput efficiency
- Fairness across lanes
- Statistical significance tests

---

## 8. Experiment Management

### 📦 Backup System

The project includes an organized backup system for tracking experiments:

```plaintext
backup/
├── exp1_10k_10bins/              # Experiment 1: 10k episodes, 10 bins
│   ├── trained_q_table.json
│   └── training_log.json
├── exp2_10k_5bins_uniform/       # Experiment 2: 10k episodes, 5 bins
│   ├── trained_q_table.json
│   └── training_log.json
└── old_training_20251029/        # Dated backup
    ├── trained_q_table.json
    └── training_log.json
```

### 🔄 Creating a New Experiment Backup

**Before starting a new experiment:**

```bash
# Create backup directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p backup/exp_${TIMESTAMP}

# Copy current results
cp trained_q_table.json backup/exp_${TIMESTAMP}/
cp training_log.json backup/exp_${TIMESTAMP}/
cp config/sim_config.json backup/exp_${TIMESTAMP}/

# Add description
echo "Experiment: [your description]" > backup/exp_${TIMESTAMP}/README.txt
```

### 📝 Experiment Naming Convention

Use descriptive names for backup folders:

- `exp1_10k_10bins` - 10,000 episodes with 10-bin discretization
- `exp2_20k_lr0.05` - 20,000 episodes with learning rate 0.05
- `exp3_custom_rewards` - Custom reward weights experiment

---

### 🗂️ Results Storage

Detailed results are automatically saved with timestamps:

```plaintext
results/
└── detailed_results_20251029_115644.json
```

**Structure of results file:**

```json
{
  "timestamp": "2025-10-29T11:56:44",
  "config": {...},
  "final_metrics": {
    "avg_reward": 42.3,
    "avg_queue": 7.8,
    "throughput": 1250
  },
  "episode_data": [...]
}
```

---

## 9. Current Status & Next Steps

### ✅ Completed Components

- ✅ Vision pipeline (YOLOv8 + DBSCAN) fully functional
- ✅ Jamming Machine simulator operational
- ✅ Q-Learning agent training complete (10,000 episodes)
- ✅ Modular reward system with configurable weights
- ✅ Comprehensive logging and analysis tools
- ✅ Proper project structure with `src/` packaging
- ✅ Experiment backup and version control system

### 🎯 Current Performance

**Training Results (Latest Run):**

- Final average reward: ~42.3 per step
- Average queue length: ~7.8 vehicles
- Performance: **Comparable to fixed-time baseline**
- Convergence: Agent learning but room for improvement

### 🔬 Next Research Steps

#### 1. Hyperparameter Optimization

- **Learning Rate Sweep:**
  ```bash
  # Test different learning rates
  for lr in 0.05 0.1 0.15 0.2; do
    # Update config/sim_config.json with new lr
    python scripts/main_rl_training.py
    # Backup results
  done
  ```
- Experiment with discount factors (0.90, 0.95, 0.99)
- Adjust reward weights to prioritize throughput

#### 2. Extended Training

- Run 20,000+ episodes to check for delayed learning
- Implement learning rate decay schedule
- Add curriculum learning (gradually increase difficulty)

#### 3. Enhanced State Representation

Consider adding:

- Temporal features (traffic trends over last N steps)
- Yellow light phase information
- Time-of-day encoding
- Historical queue patterns

#### 4. Advanced RL Algorithms

Next algorithms to implement:

- **Deep Q-Network (DQN):** Neural network approximation
- **Double DQN:** Reduced overestimation
- **Dueling DQN:** Separate value and advantage streams
- **Policy Gradients:** A3C, PPO for continuous actions

#### 5. Real-World Validation

- Test on different traffic scenarios (rush hour, night time)
- Validate on multiple intersections
- Compare with actual traffic signal timings
- Collect real-world performance metrics

---

### 📈 Known Limitations

1. **State Space Discretization:**

   - Currently uses 5-bin discretization
   - May lose important information
   - **Solution:** Try finer bins or neural network approximation

2. **Reward Function:**

   - Default weights may not be optimal
   - Trade-offs between objectives not fully explored
   - **Solution:** Systematic hyperparameter search

3. **Simulation Realism:**

   - Stochastic models are simplified
   - May not capture all real-world complexity
   - **Solution:** Calibrate with real traffic data

4. **Scalability:**

   - Q-table approach doesn't scale to complex intersections
   - Memory grows exponentially with state dimensions
   - **Solution:** Transition to DQN for larger state spaces

5. **Training Time:**
   - 10,000 episodes may not be enough for convergence
   - Need longer training or better exploration
   - **Solution:** Extended training + adaptive exploration

---

## 10. Troubleshooting

### Issue: Module Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
The project uses proper Python packaging. Ensure you're running from the project root:

```bash
cd /path/to/aitc
python scripts/main_vision_processing.py  # Not just: python main_vision_processing.py
```

The `sys.path.append()` in scripts handles the import paths correctly.

---

### Issue: YOLO Model Not Found

**Problem:** `FileNotFoundError: yolov8n.pt not found`

**Solution:**
YOLO model downloads automatically on first run. If it fails:

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads ~6MB model
```

Or download manually from: https://github.com/ultralytics/assets/releases

---

### Issue: Training Seems Stuck or Not Improving

**Diagnostic Steps:**

1. **Check learning curve:**

   ```bash
   jupyter lab
   # Open notebooks/2_results_analysis.ipynb
   # Run learning curve analysis cells
   ```

2. **Verify reward signal variability:**

   ```python
   import json
   import numpy as np
   with open('training_log.json') as f:
       log = json.load(f)
   rewards = [e['total_reward'] for e in log]
   print(f"Reward mean: {np.mean(rewards):.2f}")
   print(f"Reward std: {np.std(rewards):.2f}")
   print(f"Reward range: {np.min(rewards):.2f} to {np.max(rewards):.2f}")
   ```

   If std is very low, reward function may not be sensitive enough.

3. **Check state space coverage:**

   ```bash
   python check_state_coverage_FIXED.py
   ```

4. **Analyze Q-table statistics:**
   ```python
   import json
   with open('trained_q_table.json') as f:
       q_table = json.load(f)
   print(f"States visited: {len(q_table)}")
   print(f"Expected states: {5**8} = 390,625")  # 5 bins, 8 features
   ```

**Potential Fixes:**

- **Increase learning rate:** Try 0.15 or 0.2
- **Adjust reward weights:** Increase throughput weight
- **Extend training:** Run 20,000+ episodes
- **Improve exploration:** Slower epsilon decay
- **Better state representation:** Add more informative features

---

### Issue: Out of Memory During Training

**Problem:** Training crashes with `MemoryError`

**Solutions:**

1. **Reduce episodes per session:**

   ```json
   // In config/sim_config.json
   {
     "simulation_episodes": 5000 // Instead of 10000
   }
   ```

2. **Clear Q-table periodically** (loses learning):

   ```python
   # In src/agent/q_learning_agent.py
   if episode % 5000 == 0:
       self.q_table.clear()  # Reset Q-table
   ```

3. **Use neural network approach:**

   - Transition to DQN which uses fixed memory
   - No Q-table storage needed

4. **Save checkpoints more frequently:**
   ```python
   if episode % 1000 == 0:
       agent.save_q_table(f'models/checkpoint_{episode}.json')
   ```

---

### Issue: Vision Processing Fails

**Problem:** No vehicles detected or incorrect detections

**Diagnostic Steps:**

1. **Check image format and size:**

   ```python
   from PIL import Image
   img = Image.open('data/input_images/lane_north.jpg')
   print(f"Size: {img.size}, Mode: {img.mode}")
   ```

2. **Test YOLO directly:**

   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')
   results = model('data/input_images/lane_north.jpg')
   results[0].show()  # Display detection
   ```

3. **Adjust confidence threshold:**

   ```python
   # In src/vision/yolo_processor.py
   if score >= 0.3:  # Lower threshold from 0.5
       detections.append(...)
   ```

4. **Check DBSCAN parameters:**
   ```python
   # In scripts/main_vision_processing.py
   dbscan_analyzer = DBSCANAnalyzer(
       eps=50,      # Increase from 30
       min_samples=2
   )
   ```

---

### Issue: Results Don't Match Expected Performance

**Problem:** Agent performs worse than baseline

**Investigation:**

1. **Compare with baselines properly:**

   ```bash
   python scripts/fresh_baseline_eval.py
   jupyter lab notebooks/3_enhanced_baseline_comparison.ipynb
   ```

2. **Check reward function logic:**

   - Open `src/simulation/reward_calculator.py`
   - Verify component weights sum to 1.0
   - Ensure all components scaled correctly (0-100)

3. **Analyze episode variability:**

   ```python
   # High variance suggests unstable learning
   import numpy as np
   episode_rewards = [e['total_reward'] for e in log[-1000:]]
   print(f"Last 1000 episodes std: {np.std(episode_rewards)}")
   ```

4. **Verify state encoding:**
   ```python
   # Check discretization is reasonable
   agent = QLearningAgent(...)
   test_state = np.array([0.0, 0.0, 0.5, 0.5, 0.3, 0.2, 0.8, 0.9])
   discrete = agent._discretize_state(test_state)
   print(f"State: {test_state}")
   print(f"Discretized: {discrete}")
   ```

---

## 📚 Additional Resources

### Code Documentation

- **YOLOv8 Docs:** https://docs.ultralytics.com/
- **DBSCAN:** https://scikit-learn.org/stable/modules/clustering.html#dbscan
- **Q-Learning:** Sutton & Barto - Reinforcement Learning: An Introduction

### Useful Scripts

**Quick performance check:**

```bash
python scripts/check_rl_timing.py
```

**Comprehensive integration test:**

```bash
python scripts/final_comprehensive_test.py
```

**Fresh baseline evaluation:**

```bash
python scripts/fresh_baseline_eval.py
```

### Contact & Support

For questions or issues:

1. Check this README first
2. Review notebook analyses
3. Check `archive/debug_files/` for debugging examples
4. Open an issue in the repository

---

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@thesis{aitc2025,
  title={Traffic Control Using Artificial Intelligence For Adaptive And Optimized Traffic Management},
  author={[Your Name]},
  year={2025},
  school={[Your Institution]}
}
```

---

## 📝 License

[Specify your license here]

---

## 🙏 Acknowledgments

This project implements concepts from:

- **Computer Vision:** YOLOv8 architecture (Ultralytics)
- **Machine Learning:** DBSCAN clustering (scikit-learn)
- **Reinforcement Learning:** Q-Learning algorithm (Sutton & Barto)

Special thanks to the open-source community for the excellent tools and libraries.

---

## 🔄 Version History

**v2.0** (November 2025)

- ✅ Proper Python package structure with `src/`
- ✅ Organized scripts, config, data, and results folders
- ✅ Experiment backup system
- ✅ Enhanced documentation
- ✅ Comprehensive troubleshooting guide

**v1.0** (October 2025)

- Initial implementation
- Basic Q-Learning agent
- Vision pipeline integration
- Jamming Machine simulator

---

**Last Updated:** November 2, 2025  
**Version:** 2.0 (Corrected)  
**Status:** Active Development  
**Project Structure:** ✅ Verified Against Actual Codebase

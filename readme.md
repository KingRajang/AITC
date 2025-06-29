# AITC: AI-Powered Adaptive Traffic Control

Welcome to the **AITC project repository!** This framework is the official implementation for the thesis project:  
**"Traffic Control Using Artificial Intelligence For Adaptive And Optimized Traffic Management."**

## 1. Project Overview & Goal

The goal of this project is to design, train, and evaluate an intelligent system that can adaptively control traffic signals in real-time.  
We achieve this by integrating three distinct AI algorithms, each handling a specific part of the problem.

This README serves as the central guide for understanding the project's architecture, workflow, and how to run the experiments.

---

## 2. System Architecture & Logic

Our system is built on a **three-phase methodology** that creates a pipeline from visual data to intelligent action.  
This modular design allows us to test and improve each part of the system independently.

### 📂 Project Directory Structure

aitc/
│
├── config/              # All configuration files (lane definitions, sim params)
├── data/                # Input images, output images, and generated data files
├── notebooks/           # Jupyter notebooks for testing and analysis
├── src/                 # All Python source code for the project
│   ├── agent/           # Contains the Q-Learning agent
│   ├── simulation/      # Contains the "Jamming Machine" simulator
│   └── vision/          # Contains the YOLOv8 and DBSCAN processors
├── main_vision_processing.py  # Entrypoint script for Phase 1
└── main_rl_training.py      # Entrypoint script for Phase 2 & 3

## 🚦 Phase 1: Vision Pipeline (State Assessment)

### What it does:

This phase analyzes real-world traffic conditions from images.

### How it works:

* `main_vision_processing.py` reads lane definitions from `config/lane_config.json`.
* Loads traffic images (e.g., `lane_north.jpg`).
* `YOLOProcessor` (`src/vision/yolo_processor.py`) detects vehicles in each image.
* `DBSCANAnalyzer` (`src/vision/dbscan_analyzer.py`) clusters vehicle locations and calculates:

  * `vehicle_count`
  * `density_score`

### Output:

Creates `data/initial_state.json` describing the traffic state.

---

## 🛠 Phase 2: "Jamming Machine" (Simulation Environment)

### What it does:

A custom-built simulator that provides a fast, safe environment for training the AI agent, avoiding the computational cost of real-time vision processing.

### How it works:

* `JammingMachine` (`src/simulation/environment.py`) initializes the simulation.
* Can optionally load the initial traffic state from `initial_state.json`.
* Uses stochastic models from `src/simulation/traffic_models.py` to simulate:

  * Car arrivals
  * Car departures based on traffic light states
* Tracks vehicle counts, waiting times, and computes reward signals.

---

## 🧠 Phase 3: Q-Learning Agent (Policy Optimization)

### What it does:

The learning agent determines the best traffic light strategy in different traffic conditions.

### How it works:

* `QLearningAgent` (`src/agent/q_learning_agent.py`) interacts with the simulator over thousands of episodes.
* At each step:

  * Observes the current state
  * Chooses an action (e.g., "set North-South green")
  * Receives a reward based on the result
* Updates its internal Q-table to improve decision-making over time.

### Output:

* `trained_q_table.json` — the agent’s learned policy.
* `training_log.json` — training progress and rewards.

---

## 3. How to Set Up and Run the Project

### Step 1: Initial Setup

```bash
# Clone the repository
git clone <the-repo-url>
cd aitc

# Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> 💡 **Reminder:** Run `source venv/bin/activate` every time you open a new terminal.

---

### Step 2: Running the Full Experiment

1. **Prepare Input Images:**

   * Place your four traffic images (e.g., `lane_north.jpg`) in `data/input_images/`.

2. **Run Phase 1 (Vision Pipeline):**

   ```bash
   python main_vision_processing.py
   ```

   > This will generate `data/initial_state.json`.

3. **Run Phase 2 & 3 (Agent Training):**

   ```bash
   python main_rl_training.py
   ```

   > The agent will train for 10,000 episodes and output:
   >
   > * `trained_q_table.json`
   > * `training_log.json`

---

### Step 3: Analyzing the Results

1. Start Jupyter Lab:

   ```bash
   jupyter lab
   ```

2. Open the notebook:

   ```
   notebooks/2_results_analysis.ipynb
   ```

3. Run all cells to:

   * Load training logs
   * Compare agent performance with a baseline
   * Plot learning curves and results

---

## 4. Current Status and Next Steps

### ✅ Current Result:

* The agent has completed its first training run.
* Performance is comparable to a fixed-time controller (baseline).

### 🔬 Next Steps:

* **Analyze Learning Curves:** Does the agent actually learn, or is the reward curve flat?
* **Tune Hyperparameters:** Adjust `learning_rate`, `discount_factor`, and reward weights in `environment.py`.
* **Improve Reward Function:** Check if the reward function properly incentivizes efficient traffic flow.
* **Longer Training:** Try 20,000 or 50,000 episodes to see potential improvements.

---

Let’s get to work and make AITC even better! 🚀

```

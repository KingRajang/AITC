Status: Initial Framework Complete

1. Project Overview
Welcome to the AITC project repository! We have successfully built and tested the complete end-to-end framework for our thesis. This includes the three core phases:

Phase 1: Vision Pipeline: Uses YOLOv8 and DBSCAN to analyze traffic images and determine vehicle counts and lane density.

Phase 2: "Jamming Machine" Simulator: A custom environment where our agent can train safely and quickly.

Phase 3: Q-Learning Agent: The "brain" that learns to control the traffic signals.

2. Current Status & Next Steps
We have just completed our first full training run of 10,000 episodes.

Result: The agent currently performs similarly to a basic fixed-time controller (a -0.14% improvement in waiting time).

This is our scientific baseline. Our job now is to work together to improve this number. This is the experimental phase of our thesis.

Our immediate goal is to analyze the training data (from training_log.json) and tune the system's hyperparameters.

3. How to Set Up the Project
To get started, follow these steps in your WSL Ubuntu terminal:

Clone the repository:

git clone <the-repo-url-you-create>
cd aitc

Create a Python virtual environment:

python3 -m venv venv

Activate the virtual environment:

source venv/bin/activate

(Remember to do this every time you open a new terminal to work on the project!)

Install all required libraries:

pip install -r requirements.txt

4. How to Run the Framework
The project has two main parts you can run:

A) Test the Vision Pipeline (for one image):
This is useful for debugging our image processing code.

Start the Jupyter server: jupyter lab

Open your browser and navigate to notebooks/1_vision_pipeline_testbed.ipynb.

Run the cells to see the detection results on a single lane image.

B) Run the Full Experiment:
This runs the entire process, from analyzing all images to training the agent.

Run Phase 1 (Vision):

python main_vision_processing.py

(This creates the data/initial_state.json file)

Run Phase 2 & 3 (Training):

python main_rl_training.py

(This will take a long time. It reads initial_state.json and produces trained_q_table.json and training_log.json)

5. Let's Get to Work!
Our next tasks are:

Analyze the learning curve in 2_results_analysis.ipynb to see if the agent was actually improving.

Tune Hyperparameters: Experiment with learning_rate, discount_factor, and the reward function weights in environment.py.

Improve State Representation: Can we make the state vector more informative for the agent?

Longer Training: Try running for 20,000 or 50,000 episodes.

Let's start collaborating and get this agent trained!
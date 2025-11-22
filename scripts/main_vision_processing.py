import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import json
from src.vision.yolo_processor import YOLOProcessor
from src.vision.dbscan_analyzer import DBSCANAnalyzer

# --- Configuration ---
CONFIG_PATH = "config/lane_config.json"
INPUT_DIR = "data/input_images"
OUTPUT_DIR = "data/output_images"
STATE_OUTPUT_FILE = "data/initial_state.json"
YOLO_MODEL_PATH = "yolov8n.pt"

def main():
    """
    Entrypoint for Phase 1: Initial State Assessment.
    This script now processes a separate image for each lane defined in the config.
    """
    print("--- Phase 1: Initial State Assessment (Multi-Image) ---")

    # --- Load Configuration ---
    with open(CONFIG_PATH) as f:
        lane_config = json.load(f)
    lanes_to_process = lane_config['lanes']
    print(f"Loaded configuration for {len(lanes_to_process)} lanes.")

    # --- Initialization ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    yolo_processor = YOLOProcessor(YOLO_MODEL_PATH)
    dbscan_analyzer = DBSCANAnalyzer(eps=30, min_samples=2)

    # --- Vision Processing Pipeline (Loop through each lane) ---
    global_state = {}
    for lane_def in lanes_to_process:
        lane_id = lane_def['lane_id']
        image_name = lane_def['image_filename']
        input_path = os.path.join(INPUT_DIR, image_name)
        
        print(f"\nProcessing Lane: '{lane_id}' using image '{image_name}'...")

        if not os.path.exists(input_path):
            print(f"  [WARNING] Image not found at {input_path}. Skipping this lane.")
            global_state[lane_id] = {"vehicle_count": 0, "density_score": 0.0}
            continue

        # 1. Detect vehicles in the specific lane image
        detections, annotated_image = yolo_processor.detect_vehicles(input_path)
        print(f"  Detected {len(detections)} vehicles.")

        # 2. Analyze the detections for this single lane
        # This method is now simpler as it doesn't need to filter by polygon
        lane_state = dbscan_analyzer.analyze_single_lane(detections)
        print(f"  Analysis complete. State: {lane_state}")
        
        # 3. Add this lane's result to the global state
        global_state[lane_id] = lane_state
        
        # 4. Save the annotated image for visualization
        if annotated_image is not None:
            output_path = os.path.join(OUTPUT_DIR, f"annotated_{image_name}")
            annotated_image.save(output_path)
            print(f"  Saved annotated image to: {output_path}")

    # --- Finalize ---
    # Save the aggregated global state to a file
    with open(STATE_OUTPUT_FILE, 'w') as f:
        json.dump(global_state, f, indent=4)
    print(f"\nGlobal initial state saved to '{STATE_OUTPUT_FILE}'")
    print(json.dumps(global_state, indent=4))
    print("\n--- Phase 1 Complete ---")

if __name__ == "__main__":
    main()
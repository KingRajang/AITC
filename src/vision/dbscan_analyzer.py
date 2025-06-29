import numpy as np
from sklearn.cluster import DBSCAN

class DBSCANAnalyzer:
    """
    Performs per-lane vehicle analysis using DBSCAN.
    This version is updated for a multi-image setup where each image
    corresponds to a single lane.
    """
    def __init__(self, eps: int, min_samples: int):
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        print(f"DBSCANAnalyzer initialized with eps={eps}, min_samples={min_samples}")

    def _get_vehicle_position(self, detection: dict) -> tuple:
        box = detection['box']
        x_center = (box[0] + box[2]) / 2
        y_bottom = box[3]
        return (int(x_center), int(y_bottom))

    def _classify_density(self, vehicle_count: int, cluster_count: int) -> float:
        if vehicle_count == 0:
            return 0.0  # "Empty"
        if cluster_count == 1 and vehicle_count > 3:
            return 0.9 # "Congested"
        elif cluster_count > 3:
            return 0.5 # "Irregular"
        else:
            return 0.3 # "Smooth"

    def analyze_single_lane(self, detections: list) -> dict:
        """
        Analyzes vehicle detections from a single lane's image.
        Since the image is for one lane, all detections belong to it.
        No polygon filtering is needed.

        Args:
            detections (list): A list of vehicle detections from YOLOProcessor.

        Returns:
            dict: A dictionary containing the vehicle count and density score for the lane.
        """
        vehicle_count = len(detections)
        
        # If there are vehicles, perform DBSCAN
        if vehicle_count > 0:
            positions = [self._get_vehicle_position(det) for det in detections]
            self.dbscan.fit(np.array(positions))
            labels = self.dbscan.labels_
            # Number of clusters found, ignoring noise points (label -1)
            cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
        else:
            cluster_count = 0

        # Classify density based on the results
        density_score = self._classify_density(vehicle_count, cluster_count)

        # Return the final state for this lane
        return {
            "vehicle_count": vehicle_count,
            "density_score": density_score
        }
from ultralytics import YOLO
from PIL import Image
import cv2

class YOLOProcessor:
    """
    Handles vehicle detection using the YOLOv8 model, as described in
    Section 3.3.1 of the thesis paper.
    """
    def __init__(self, model_path: str):
        """
        Initializes the YOLOv8 model.

        Args:
            model_path (str): The path to the YOLOv8 model file (.pt).
        """
        # Load the specified YOLOv8 model. 'yolov8n.pt' is recommended for its speed.
        self.model = YOLO(model_path)
        
        # Per Section 3.3.1, the target vehicle classes from the COCO dataset are:
        # car (2), motorcycle (3), bus (5), and truck (7).
        self.vehicle_classes = [2, 3, 5, 7]
        print("YOLOv8 Processor initialized for vehicle detection.")

    def detect_vehicles(self, image_path: str):
        """
        Detects vehicles in a given image and returns detections and an annotated image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            A tuple containing:
            - list: A list of detections. Each detection is a dictionary:
                    {'box': [x1, y1, x2, y2], 'class_id': int, 'score': float}
            - Image: A PIL Image object of the annotated frame for visualization.
        """
        try:
            # Run inference on the image. The model returns a list of Results objects.
            results = self.model(image_path, verbose=False) # verbose=False for cleaner output
            
            # Use the plot() method from ultralytics to get an annotated frame (BGR numpy array)
            annotated_frame_bgr = results[0].plot()
            
            # Convert the BGR numpy array to an RGB PIL Image for standard handling
            annotated_image = Image.fromarray(cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB))
            
            detections = []
            # The result for the first image is at index 0
            for box in results[0].boxes:
                class_id = int(box.cls)
                
                # Filter for only the specified vehicle classes
                if class_id in self.vehicle_classes:
                    # Get bounding box coordinates [x1, y1, x2, y2]
                    coords = box.xyxy[0].tolist()
                    # Get the confidence score
                    score = float(box.conf)
                    
                    # Per Section 3.3.1, use a confidence threshold of 0.5
                    if score >= 0.5:
                        detections.append({
                            'box': coords,
                            'class_id': class_id,
                            'score': score
                        })
            
            return detections, annotated_image

        except Exception as e:
            print(f"An error occurred during YOLOv8 detection: {e}")
            return [], None
            
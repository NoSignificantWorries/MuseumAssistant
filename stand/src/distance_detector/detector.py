import os
from pathlib import Path
from collections import deque
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


class InteractiveStandDetector:
    """
    Detector for a single person in front of the stand and rough distance estimation.

    Uses a YOLOv8 Pose model to detect human keypoints on video frames.
    The distance is estimated from the bounding box height when the model
    confidently sees the head (nose) and at least one shoulder.

    Attributes:
        model_path (Path): Local path to the YOLOv8 pose model file.
        model (YOLO): Loaded YOLOv8 model used for pose detection.
        activation_distance (float): Distance threshold (in meters) used by the stand logic.
        speed_history (deque[float]): History of horizontal center displacements
            for step‑slowdown analysis.
        last_center (tuple[int, int] | None): Last known center of the person bbox.
        slowing_down (bool): Flag indicating that the person is slowing down.
    """

    def __init__(self):

        self.model_path = Path("../shared/yolo_weights/yolov8n-pose.pt").resolve().absolute()

        print(self.model_path, self.model_path.exists())
    
        # Create directory for the model if it does not exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # If there is no local model file – download and save it
        if not os.path.exists(self.model_path):
            print("Downloading YOLOv8 Pose...")
            temp_model = YOLO('yolov8n-pose.pt')  # Downloaded into Ultralytics cache
            temp_model.save(self.model_path)      # Save to local folder
            print(f"Model saved to: {self.model_path}")
        else:
            print(f"Local model found: {self.model_path}")
        
        # Load local model
        self.model = YOLO(self.model_path)
        self.activation_distance = 2.0
        
        # Step‑slowdown analysis
        self.speed_history = deque(maxlen=10)  # Last 10 speed values
        self.last_center = None
        self.slowing_down = False
        
        print("ДDetector is ready.")

    def _analyze_speed(self, center):
        """
        Estimate whether the person in front of the stand is slowing down.

        Tracks the horizontal movement of the bounding box center over time and
        computes the average speed in pixels per frame. If the average speed
        over the last N frames drops below a threshold, the detector considers
        that the person is slowing down.

        Args:
            center (tuple[int, int]): Current bounding box center (x, y).

        Returns:
            bool: True if the person is likely slowing down, False otherwise.
        """
        if self.last_center is None:
            self.last_center = center
            return False
        
        # Speed in pixels per frame (assuming ~30 FPS)
        dx = abs(center[0] - self.last_center[0])
        self.speed_history.append(dx)
        
        if len(self.speed_history) >= 10:
            avg_speed = np.mean(list(self.speed_history)[-5:])
            self.last_center = center
            
            # Slowdown: speed < 0.8 pixels per frame
            if avg_speed < 0.8:
                self.slowing_down = True
                return True
            else:
                self.slowing_down = False
        
        return False

    def get_person_depth(self, frame) -> Optional[tuple[float, np.array]]:
        """
        Analyze a video frame and estimate the distance to a person, if detected.

        The method runs YOLOv8 Pose on the frame, checks confidence for the head
        (nose) and at least one shoulder, estimates distance from the bounding
        box height, draws the bbox on the frame and optionally marks slowdown
        state. If there is no reliable pose, it returns None.

        Args:
            frame (np.ndarray): Input video frame in BGR format (OpenCV).

        Returns:
            Optional[tuple[float, np.ndarray]]:
                - (distance_in_meters, nose_keypoint_xy) if a confident person
                  with head and shoulder keypoints is detected;
                - None if no person is found or keypoints are not reliable.
        """
        results = self.model(frame, verbose=False)
        
        # If no people are detected – return None
        if len(results[0].boxes) == 0:
            self.last_center = None  # Reset speed tracking
            return None
        
        keypoints = results[0].keypoints
        
        # Если нет ключевых точек — fallback на bbox без проверки pose
        # if len(keypoints) == 0:
        #     box = results[0].boxes.xyxy[0].cpu().numpy()
        #     box_height = box[3] - box[1]
        #     distance = max(0.5, 300 / box_height)
        #     
        #     # Анализ скорости даже без позы
        #     center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        #     slowing_down = self._analyze_speed(center)
        #     
        #     x1, y1, x2, y2 = map(int, box)
        #     color = (0, 255, 0) if slowing_down else (0, 255, 255)
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        #     return distance, keypoints.xy[0].cpu().numpy()[0]
        
        # Get confidence scores for keypoints of the first detection
        conf = keypoints.conf[0].cpu().numpy()
        
        # YOLOv8 Pose keypoint indices:
        # 0 - nose (head)
        # 5 - left shoulder
        # 6 - right shoulder
        nose_conf = conf[0]
        left_shoulder_conf = conf[5]
        right_shoulder_conf = conf[6]
        
        # Confidence threshold for head and shoulders
        threshold = 0.5
        
        # Check that the head and at least one shoulder are confident enough
        if nose_conf > threshold and (left_shoulder_conf > threshold or right_shoulder_conf > threshold):
            box = results[0].boxes.xyxy[0].cpu().numpy()
            box_height = box[3] - box[1]
            distance = max(0.5, 300 / box_height)
            
            # Analyze speed for this bbox center
            center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            slowing_down = self._analyze_speed(center)
            
            x1, y1, x2, y2 = map(int, box)
            # Green if we have a confident person + slowdown, yellow otherwise
            color = (0, 255, 0) if slowing_down else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            return distance, keypoints.xy[0].cpu().numpy()[0]
        else:
            # If data is not confident enough – draw red bbox and return None
            box = results[0].boxes.xyxy[0].cpu().numpy()
            center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            self._analyze_speed(center)  # Still track speed
            
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            return None

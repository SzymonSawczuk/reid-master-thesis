"""
Detection Module

Handles object detection using YOLO.
Separates detections into person and vehicle categories.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import torch
from ultralytics import YOLO
from .utils import crop_bbox


class DetectionModule:
    """
    Detection module with YOLO for person and vehicle detection.
    """

    def __init__(
        self,
        model_name: str = 'yolov8x.pt',
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = 'cuda'
    ):
        """
        Initialize detection module.

        Args:
            model_name: YOLO model name (e.g., 'yolov8x.pt')
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        
        print(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        if torch.cuda.is_available() and device == 'cuda':
            self.model.to(device)

        # COCO class IDs
        self.PERSON_CLASS = 0
        self.VEHICLE_CLASSES = [2, 3, 5, 7]  

        
        self.detection_counter = 0

        print(f"Detection module initialized (device: {device})")

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0
    ) -> Dict[str, List[Dict]]:
        """
        Detect objects in a frame.

        Args:
            frame: Input frame (BGR format)
            frame_id: Frame number (kept for compatibility)

        Returns:
            Dictionary with 'persons' and 'vehicles' lists, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: detection confidence
                - detection_id: unique detection ID
                - cropped_image: cropped bounding box image
        """
        
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]

        
        boxes = results.boxes.xyxy.cpu().numpy()  
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        
        person_detections = []
        vehicle_detections = []

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            
            cropped = crop_bbox(frame, box)

            
            detection_obj = {
                'bbox': box.tolist(),
                'confidence': float(conf),
                'detection_id': self.detection_counter,
                'cropped_image': cropped,
                'frame_id': frame_id
            }
            self.detection_counter += 1

            if cls_id == self.PERSON_CLASS:
                person_detections.append(detection_obj)
            elif cls_id in self.VEHICLE_CLASSES:
                vehicle_detections.append(detection_obj)

        return {
            'persons': person_detections,
            'vehicles': vehicle_detections
        }

    def reset_counter(self):
        """Reset detection counter."""
        self.detection_counter = 0

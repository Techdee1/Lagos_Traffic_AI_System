"""
Lagos Vehicle Detector
Detects and classifies Lagos-specific vehicle types using pretrained YOLOv8 + post-processing
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging
from pathlib import Path

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    MODEL_NAME, CONFIDENCE_THRESHOLD, INPUT_SIZE, DEVICE,
    COCO_VEHICLE_CLASSES, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER,
    YELLOW_RATIO_THRESHOLD, BRT_ASPECT_RATIO_THRESHOLD,
    KEKE_WIDTH_MIN, KEKE_WIDTH_MAX, KEKE_HEIGHT_RATIO_MIN,
    KEKE_YELLOW_THRESHOLD, BLUE_HSV_LOWER, BLUE_HSV_UPPER,
    BLUE_RATIO_THRESHOLD
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LagosVehicleDetector:
    """
    Detects and classifies Lagos vehicles:
    - Okada (motorcycles)
    - Keke Napep (tricycles)
    - Danfo (yellow minibuses)
    - BRT (long buses)
    - Private cars
    - Trucks/Trailers
    """
    
    def __init__(self, model_path=None, confidence=None, device=None):
        """Initialize the detector with pretrained YOLOv8 model"""
        self.model_path = model_path or MODEL_NAME
        self.confidence = confidence or CONFIDENCE_THRESHOLD
        self.device = device or DEVICE
        
        logger.info(f"Loading YOLOv8 model: {self.model_path}")
        self.model = YOLO(self.model_path)
        logger.info(f"Model loaded successfully on {self.device}")
        
        # COCO vehicle class IDs we care about
        self.vehicle_classes = COCO_VEHICLE_CLASSES
        
    def detect(self, frame):
        """
        Run detection on frame and classify Lagos vehicle types
        
        Args:
            frame: OpenCV image (BGR)
            
        Returns:
            list of dicts with keys: class_name, confidence, bbox, vehicle_type
        """
        # Run YOLO detection
        results = self.model(
            frame,
            conf=self.confidence,
            imgsz=INPUT_SIZE,
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                class_id = int(box.cls[0])
                
                # Filter only vehicle classes
                if class_id not in self.vehicle_classes:
                    continue
                
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = [x1, y1, x2, y2]
                
                # Get base vehicle type from COCO
                base_type = self.vehicle_classes[class_id]
                
                # Classify to Lagos-specific types
                vehicle_type = self._classify_lagos_vehicle(
                    frame, bbox, base_type, confidence
                )
                
                detection = {
                    'class_id': class_id,
                    'class_name': base_type,
                    'confidence': confidence,
                    'bbox': bbox,
                    'vehicle_type': vehicle_type
                }
                
                detections.append(detection)
        
        return detections
    
    def _classify_lagos_vehicle(self, frame, bbox, base_type, confidence):
        """
        Classify detected vehicle to Lagos-specific type
        
        Args:
            frame: OpenCV image
            bbox: [x1, y1, x2, y2]
            base_type: COCO class name (motorcycle, car, bus, truck)
            confidence: Detection confidence score
            
        Returns:
            str: Lagos vehicle type
        """
        x1, y1, x2, y2 = bbox
        
        # Extract vehicle region
        vehicle_roi = frame[y1:y2, x1:x2]
        
        if vehicle_roi.size == 0:
            return base_type
        
        # Calculate dimensions
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 0
        
        # Motorcycle -> Check if it's actually a keke (tricycle)
        if base_type == "motorcycle":
            # Keke are yellow three-wheelers - wider, enclosed, and boxy
            if self._is_keke_napep(vehicle_roi, width, height):
                return "keke_napep"
            return "okada"
        
        # Car -> Check if it's a keke (three-wheeler detected as small car)
        elif base_type == "private_car":
            # Low confidence cars that are yellow/black = likely keke
            # Keke often gets misclassified as car with lower confidence
            if confidence < 0.5 and self._has_yellow_black_pattern(vehicle_roi):
                return "keke_napep"
            # Also check standard keke characteristics
            if self._is_keke_napep(vehicle_roi, width, height):
                return "keke_napep"
            return "private_car"
        
        # Bus -> Check if danfo (yellow) or BRT (long)
        elif base_type == "bus":
            return self._classify_bus(vehicle_roi, width, height, aspect_ratio)
        
        # Truck -> Direct mapping
        elif base_type == "truck":
            return "truck"
        
        return base_type
    
    def _classify_bus(self, vehicle_roi, width, height, aspect_ratio):
        """
        Classify bus as danfo or BRT
        
        BRT buses in Lagos are:
        - Large and long (high aspect ratio)
        - Blue in color
        
        Danfo buses are:
        - Yellow minibuses
        
        Args:
            vehicle_roi: Cropped image of vehicle
            width, height: Bounding box dimensions
            aspect_ratio: Width/height ratio
            
        Returns:
            str: 'danfo', 'brt', or 'bus'
        """
        # Check if yellow (danfo)
        if self._is_yellow_dominant(vehicle_roi):
            return "danfo"
        
        # Check if blue (BRT) - BRT buses are blue in Lagos
        if self._is_blue_dominant(vehicle_roi):
            return "brt"
        
        # Check if long bus (BRT) - even without blue, very long buses are BRT
        if self._is_long_vehicle(aspect_ratio) and width > 200:
            return "brt"
        
        # Generic bus
        return "bus"
    
    def _is_yellow_dominant(self, vehicle_roi):
        """
        Check if vehicle is predominantly yellow (for danfo detection)
        
        Args:
            vehicle_roi: Cropped BGR image of vehicle
            
        Returns:
            bool: True if yellow dominant
        """
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)
            
            # Define yellow range in HSV
            lower_yellow = np.array(YELLOW_HSV_LOWER)
            upper_yellow = np.array(YELLOW_HSV_UPPER)
            
            # Create mask for yellow pixels
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Calculate ratio of yellow pixels
            yellow_pixels = cv2.countNonZero(yellow_mask)
            total_pixels = vehicle_roi.shape[0] * vehicle_roi.shape[1]
            
            yellow_ratio = yellow_pixels / total_pixels if total_pixels > 0 else 0
            
            # Return True if significant yellow presence
            return yellow_ratio > YELLOW_RATIO_THRESHOLD
            
        except Exception as e:
            logger.error(f"Error in yellow detection: {e}")
            return False
    
    def _is_long_vehicle(self, aspect_ratio):
        """
        Check if vehicle is long (for BRT detection)
        
        Args:
            aspect_ratio: Width/height ratio
            
        Returns:
            bool: True if long vehicle (BRT)
        """
        return aspect_ratio > BRT_ASPECT_RATIO_THRESHOLD
    
    def _is_blue_dominant(self, vehicle_roi):
        """
        Check if vehicle is predominantly blue (for BRT bus detection)
        BRT buses in Lagos are blue colored
        
        Args:
            vehicle_roi: Cropped BGR image of vehicle
            
        Returns:
            bool: True if blue dominant
        """
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)
            
            # Define blue range in HSV
            lower_blue = np.array(BLUE_HSV_LOWER)
            upper_blue = np.array(BLUE_HSV_UPPER)
            
            # Create mask for blue pixels
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Calculate ratio of blue pixels
            blue_pixels = cv2.countNonZero(blue_mask)
            total_pixels = vehicle_roi.shape[0] * vehicle_roi.shape[1]
            
            blue_ratio = blue_pixels / total_pixels if total_pixels > 0 else 0
            
            # Return True if significant blue presence
            return blue_ratio > BLUE_RATIO_THRESHOLD
            
        except Exception as e:
            logger.error(f"Error in blue detection: {e}")
            return False
    
    def _has_yellow_black_pattern(self, vehicle_roi):
        """
        Check if vehicle has yellow and black color pattern (keke napep characteristic)
        
        Keke napep typically has yellow body with black stripes/trim
        
        Args:
            vehicle_roi: Cropped BGR image of vehicle
            
        Returns:
            bool: True if yellow-black pattern detected
        """
        try:
            hsv = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)
            
            # Check for yellow
            lower_yellow = np.array(YELLOW_HSV_LOWER)
            upper_yellow = np.array(YELLOW_HSV_UPPER)
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Check for black (low value in HSV)
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 50])
            black_mask = cv2.inRange(hsv, lower_black, upper_black)
            
            total_pixels = vehicle_roi.shape[0] * vehicle_roi.shape[1]
            if total_pixels == 0:
                return False
            
            yellow_ratio = cv2.countNonZero(yellow_mask) / total_pixels
            black_ratio = cv2.countNonZero(black_mask) / total_pixels
            
            # Keke has both yellow (body) and black (stripes/trim)
            # Lower thresholds for detection
            return yellow_ratio > 0.08 and black_ratio > 0.05
            
        except Exception as e:
            logger.error(f"Error in yellow-black pattern detection: {e}")
            return False
    
    def _is_keke_napep(self, vehicle_roi, width, height):
        """
        Check if vehicle is a keke napep (yellow three-wheeler tricycle)
        
        Keke characteristics:
        - Yellow with black stripes
        - Enclosed cabin, boxy shape
        - Smaller than regular cars
        
        Args:
            vehicle_roi: Cropped BGR image of vehicle
            width, height: Bounding box dimensions
            
        Returns:
            bool: True if keke napep
        """
        try:
            # Skip very large vehicles (definitely cars/trucks)
            if width > KEKE_WIDTH_MAX:
                return False
            
            # Check for yellow color (primary indicator for keke)
            hsv = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array(YELLOW_HSV_LOWER)
            upper_yellow = np.array(YELLOW_HSV_UPPER)
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            yellow_pixels = cv2.countNonZero(yellow_mask)
            total_pixels = vehicle_roi.shape[0] * vehicle_roi.shape[1]
            yellow_ratio = yellow_pixels / total_pixels if total_pixels > 0 else 0
            
            # If has yellow and is small/medium sized, it's likely keke
            if yellow_ratio > KEKE_YELLOW_THRESHOLD:
                # Additional check: keke tends to be smaller than regular cars
                # At 1080p, keke typically under 300px wide
                if width < KEKE_WIDTH_MAX:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in keke detection: {e}")
            return False
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: OpenCV image
            detections: List of detection dicts
            
        Returns:
            frame with annotations
        """
        annotated_frame = frame.copy()
        
        # Color mapping for vehicle types
        colors = {
            'okada': (0, 255, 255),      # Yellow
            'keke_napep': (255, 165, 0),  # Orange
            'danfo': (0, 255, 0),         # Green
            'brt': (255, 0, 0),           # Blue
            'private_car': (255, 255, 255), # White
            'truck': (128, 0, 128),       # Purple
            'bus': (0, 165, 255)          # Orange-red
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            vehicle_type = det['vehicle_type']
            confidence = det['confidence']
            
            # Get color
            color = colors.get(vehicle_type, (0, 255, 0))
            
            # Draw box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Format label
            label = f"{vehicle_type.upper()} {confidence:.2f}"
            
            # Draw label background (larger, more visible)
            font_scale = 0.7
            font_thickness = 2
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness
            )
            
            # Draw solid black background for better visibility
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_h - 15),
                (x1 + label_w + 10, y1),
                (0, 0, 0),
                -1
            )
            
            # Draw colored top border
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_h - 15),
                (x1 + label_w + 10, y1 - label_h - 12),
                color,
                -1
            )
            
            # Draw label text (white for maximum contrast)
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_DUPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )
        
        return annotated_frame

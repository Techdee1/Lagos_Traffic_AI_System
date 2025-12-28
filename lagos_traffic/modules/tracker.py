"""
Vehicle Tracker Module
Tracks vehicles across frames to count unique vehicles only once
Uses IoU (Intersection over Union) for matching detections
"""

import numpy as np
from collections import defaultdict
import logging
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VehicleTracker:
    """
    Tracks vehicles across frames using IoU matching
    Ensures each unique vehicle is counted only once
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_frames_missing: int = 30,
        min_hits: int = 3
    ):
        """
        Initialize tracker
        
        Args:
            iou_threshold: Minimum IoU to consider same vehicle
            max_frames_missing: Frames before track is deleted
            min_hits: Minimum detections before counting as valid
        """
        self.iou_threshold = iou_threshold
        self.max_frames_missing = max_frames_missing
        self.min_hits = min_hits
        
        # Track storage
        self.tracks = {}  # track_id -> track_info
        self.next_track_id = 1
        
        # Unique vehicle counts (only counted once per track)
        self.unique_counts = defaultdict(int)
        
        # Set of track IDs that have been counted
        self.counted_tracks = set()
        
        logger.info(f"VehicleTracker initialized (IoU={iou_threshold}, max_missing={max_frames_missing})")
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes
        
        Args:
            box1, box2: [x1, y1, x2, y2] format
            
        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_centroid_distance(self, box1: List[int], box2: List[int]) -> float:
        """
        Calculate distance between centroids of two boxes
        
        Args:
            box1, box2: [x1, y1, x2, y2] format
            
        Returns:
            Euclidean distance between centroids
        """
        cx1 = (box1[0] + box1[2]) / 2
        cy1 = (box1[1] + box1[3]) / 2
        cx2 = (box2[0] + box2[2]) / 2
        cy2 = (box2[1] + box2[3]) / 2
        
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    def update(self, detections: List[Dict], frame_id: int) -> Tuple[List[Dict], List[Dict]]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dicts with 'bbox', 'vehicle_type', etc.
            frame_id: Current frame number
            
        Returns:
            Tuple of (all_tracked_detections, new_unique_vehicles)
            - all_tracked_detections: Detections with track_id added
            - new_unique_vehicles: Only vehicles being counted for first time
        """
        if not detections:
            # Age all existing tracks
            self._age_tracks(frame_id)
            return [], []
        
        # Match detections to existing tracks
        matched_detections = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        
        # Create cost matrix based on IoU
        if self.tracks:
            for det_idx, det in enumerate(detections):
                det_bbox = det['bbox']
                best_iou = 0
                best_track_id = None
                
                for track_id in list(unmatched_tracks):
                    track = self.tracks[track_id]
                    track_bbox = track['bbox']
                    
                    iou = self._calculate_iou(det_bbox, track_bbox)
                    
                    # Also check vehicle type matches
                    if det['vehicle_type'] == track['vehicle_type'] and iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id
                
                if best_iou >= self.iou_threshold and best_track_id is not None:
                    # Match found
                    matched_detections.append((det_idx, best_track_id))
                    if det_idx in unmatched_detections:
                        unmatched_detections.remove(det_idx)
                    if best_track_id in unmatched_tracks:
                        unmatched_tracks.remove(best_track_id)
        
        # Update matched tracks
        for det_idx, track_id in matched_detections:
            det = detections[det_idx]
            self.tracks[track_id]['bbox'] = det['bbox']
            self.tracks[track_id]['confidence'] = det['confidence']
            self.tracks[track_id]['hits'] += 1
            self.tracks[track_id]['frames_missing'] = 0
            self.tracks[track_id]['last_frame'] = frame_id
            det['track_id'] = track_id
        
        # Create new tracks for unmatched detections
        new_unique_vehicles = []
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            track_id = self._create_track(det, frame_id)
            det['track_id'] = track_id
        
        # Age unmatched tracks
        for track_id in unmatched_tracks:
            self.tracks[track_id]['frames_missing'] += 1
        
        # Remove old tracks
        self._cleanup_tracks()
        
        # Check for new unique vehicles to count
        for det in detections:
            track_id = det.get('track_id')
            if track_id and track_id in self.tracks:
                track = self.tracks[track_id]
                # Count only if track has enough hits and hasn't been counted
                if track['hits'] >= self.min_hits and track_id not in self.counted_tracks:
                    self.counted_tracks.add(track_id)
                    self.unique_counts[det['vehicle_type']] += 1
                    det['is_new_count'] = True
                    new_unique_vehicles.append(det)
                else:
                    det['is_new_count'] = False
        
        return detections, new_unique_vehicles
    
    def _create_track(self, detection: Dict, frame_id: int) -> int:
        """Create new track for detection"""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        self.tracks[track_id] = {
            'bbox': detection['bbox'],
            'vehicle_type': detection['vehicle_type'],
            'confidence': detection['confidence'],
            'first_frame': frame_id,
            'last_frame': frame_id,
            'hits': 1,
            'frames_missing': 0
        }
        
        return track_id
    
    def _age_tracks(self, frame_id: int):
        """Age all tracks when no detections"""
        for track_id in self.tracks:
            self.tracks[track_id]['frames_missing'] += 1
        self._cleanup_tracks()
    
    def _cleanup_tracks(self):
        """Remove tracks that have been missing too long"""
        to_remove = []
        for track_id, track in self.tracks.items():
            if track['frames_missing'] > self.max_frames_missing:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def get_unique_counts(self) -> Dict[str, int]:
        """Get unique vehicle counts by type"""
        return dict(self.unique_counts)
    
    def get_active_tracks(self) -> int:
        """Get number of currently active tracks"""
        return len(self.tracks)
    
    def reset(self):
        """Reset all tracking state"""
        self.tracks = {}
        self.next_track_id = 1
        self.unique_counts = defaultdict(int)
        self.counted_tracks = set()
        logger.info("Tracker reset")
    
    def get_stats(self) -> Dict:
        """Get tracker statistics"""
        return {
            'active_tracks': len(self.tracks),
            'total_tracks_created': self.next_track_id - 1,
            'unique_counts': dict(self.unique_counts),
            'total_unique_vehicles': sum(self.unique_counts.values())
        }

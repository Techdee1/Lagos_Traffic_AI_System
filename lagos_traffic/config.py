"""
Configuration settings for Lagos Traffic Analysis System
Detects and classifies Lagos-specific vehicle types:
- Okada (motorcycles)
- Keke Napep (yellow tricycles)
- Danfo (yellow minibuses)
- BRT (long buses)
- Private vehicles (cars)
- Trucks/Trailers
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
MODULES_DIR = PROJECT_ROOT / "modules"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
TEST_VIDEOS_DIR = PROJECT_ROOT / "test_videos"
LOGS_DIR = PROJECT_ROOT / "logs"
DB_DIR = PROJECT_ROOT / "database"

# Model settings
MODEL_NAME = "yolov8n.pt"  # Pretrained COCO model (fastest)
CONFIDENCE_THRESHOLD = 0.35  # Slightly higher to reduce false positives
INPUT_SIZE = 416  # Smaller for faster CPU inference
DEVICE = "cpu"  # or "cuda" if GPU available

# Performance settings
PROCESSING_WIDTH = 640  # Resize frames for faster processing
STREAM_WIDTH = 960  # Resize for streaming (lower bandwidth)
SKIP_FRAMES = 5  # Process every 6th frame (higher = faster, less accurate)
JPEG_QUALITY = 60  # Lower = faster encoding, smaller size

# Vehicle classification settings
YELLOW_HSV_LOWER = (20, 100, 100)  # Lower bound for yellow color
YELLOW_HSV_UPPER = (30, 255, 255)  # Upper bound for yellow color
YELLOW_RATIO_THRESHOLD = 0.15  # 15% yellow pixels to classify as danfo
BRT_ASPECT_RATIO_THRESHOLD = 2.0  # Width/height ratio for BRT buses (lowered)

# BRT Bus detection (blue colored long buses)
BLUE_HSV_LOWER = (100, 50, 50)   # Lower bound for blue color
BLUE_HSV_UPPER = (130, 255, 255) # Upper bound for blue color
BLUE_RATIO_THRESHOLD = 0.10  # 10% blue pixels to classify as BRT

# Keke Napep detection (yellow three-wheeler tricycle)
# Keke is smaller than cars, yellow/black, and has boxy shape
KEKE_WIDTH_MIN = 30  # Min width in pixels (smaller threshold)
KEKE_WIDTH_MAX = 300  # Max width (increased for high-res videos)
KEKE_HEIGHT_RATIO_MIN = 0.6  # Lowered - keke can appear wider at angles
KEKE_YELLOW_THRESHOLD = 0.05  # 5% yellow (lowered - black stripes reduce yellow)

# COCO class mapping to Lagos vehicle types
COCO_VEHICLE_CLASSES = {
    2: "private_car",      # car
    3: "motorcycle",       # motorcycle -> okada
    5: "bus",              # bus -> needs yellow/size check
    7: "truck",            # truck
}

# Dashboard settings
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8081
STREAM_FPS = 8  # Lower FPS for smoother streaming on limited resources

# Database settings
DB_NAME = "lagos_traffic.db"
DB_PATH = DB_DIR / DB_NAME

# Create directories
LOGS_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

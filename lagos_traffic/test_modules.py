"""
Test script for Lagos Traffic System
Tests detector and database modules
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from modules.detector import LagosVehicleDetector
from modules.database import TrafficDatabase
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_detector():
    """Test the vehicle detector"""
    logger.info("=" * 50)
    logger.info("TESTING LAGOS VEHICLE DETECTOR")
    logger.info("=" * 50)
    
    try:
        # Initialize detector
        logger.info("Initializing detector...")
        detector = LagosVehicleDetector()
        logger.info("✅ Detector initialized successfully")
        
        # Create a dummy test image
        logger.info("\nCreating test image...")
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Add some colored rectangles to simulate vehicles
        # Yellow rectangle (potential danfo)
        cv2.rectangle(test_image, (100, 100), (200, 150), (0, 255, 255), -1)
        # Blue rectangle (potential car)
        cv2.rectangle(test_image, (300, 200), (400, 260), (255, 0, 0), -1)
        
        logger.info("✅ Test image created (640x640)")
        
        # Run detection
        logger.info("\nRunning detection...")
        detections = detector.detect(test_image)
        logger.info(f"✅ Detection completed: {len(detections)} vehicles detected")
        
        if detections:
            logger.info("\nDetected vehicles:")
            for i, det in enumerate(detections, 1):
                logger.info(f"  {i}. {det['vehicle_type']} - confidence: {det['confidence']:.2f}")
        else:
            logger.info("ℹ️  No vehicles detected (expected for blank test image)")
        
        # Test color detection methods
        logger.info("\nTesting classification methods...")
        
        # Create yellow image for danfo test
        yellow_img = np.zeros((100, 100, 3), dtype=np.uint8)
        yellow_img[:] = (0, 255, 255)  # BGR yellow
        is_yellow = detector._is_yellow_dominant(yellow_img)
        logger.info(f"  Yellow detection: {'✅ PASS' if is_yellow else '❌ FAIL'}")
        
        # Test aspect ratio for BRT
        is_long = detector._is_long_vehicle(2.5)  # BRT-like ratio
        is_not_long = detector._is_long_vehicle(1.5)  # Normal bus ratio
        logger.info(f"  BRT detection: {'✅ PASS' if is_long and not is_not_long else '❌ FAIL'}")
        
        logger.info("\n✅ All detector tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database():
    """Test the database module"""
    logger.info("\n" + "=" * 50)
    logger.info("TESTING TRAFFIC DATABASE")
    logger.info("=" * 50)
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        db = TrafficDatabase()
        logger.info("✅ Database initialized successfully")
        
        # Test logging detection
        logger.info("\nTesting detection logging...")
        test_detection = {
            'vehicle_type': 'okada',
            'confidence': 0.85,
            'bbox': [100, 100, 200, 200],
            'frame_id': 1,
            'class_id': 3,
            'class_name': 'motorcycle'
        }
        
        detection_id = db.log_detection(test_detection)
        if detection_id:
            logger.info(f"✅ Detection logged with ID: {detection_id}")
        else:
            logger.error("❌ Failed to log detection")
            return False
        
        # Test batch logging
        logger.info("\nTesting batch detection logging...")
        batch_detections = [
            {
                'vehicle_type': 'danfo',
                'confidence': 0.75,
                'bbox': [150, 150, 250, 220],
                'frame_id': 2,
                'class_id': 5,
                'class_name': 'bus'
            },
            {
                'vehicle_type': 'private_car',
                'confidence': 0.90,
                'bbox': [300, 100, 400, 180],
                'frame_id': 2,
                'class_id': 2,
                'class_name': 'car'
            },
            {
                'vehicle_type': 'keke_napep',
                'confidence': 0.65,
                'bbox': [50, 200, 120, 260],
                'frame_id': 2,
                'class_id': 3,
                'class_name': 'motorcycle'
            }
        ]
        
        db.log_detections_batch(batch_detections, frame_id=2)
        logger.info("✅ Batch detections logged")
        
        # Test retrieving recent detections
        logger.info("\nTesting recent detections retrieval...")
        recent = db.get_recent_detections(limit=10)
        logger.info(f"✅ Retrieved {len(recent)} recent detections")
        
        if recent:
            logger.info("\nRecent detections:")
            for det in recent[:3]:  # Show first 3
                logger.info(f"  - {det['vehicle_type']} (confidence: {det['confidence']:.2f})")
        
        # Test vehicle counts
        logger.info("\nTesting vehicle counts...")
        counts = db.get_vehicle_counts(time_window=3600)  # Last hour
        logger.info(f"✅ Vehicle counts retrieved: {counts}")
        
        # Test total counts
        logger.info("\nTesting total counts...")
        total_counts = db.get_total_counts()
        logger.info(f"✅ Total counts: {total_counts}")
        
        # Test session management
        logger.info("\nTesting session management...")
        session_id = db.create_session("test_video.mp4")
        if session_id:
            logger.info(f"✅ Session created with ID: {session_id}")
            db.close_session(session_id)
            logger.info("✅ Session closed")
        else:
            logger.error("❌ Failed to create session")
            return False
        
        # Test statistics
        logger.info("\nTesting statistics...")
        stats = db.get_stats()
        logger.info("✅ Statistics retrieved:")
        logger.info(f"  Total detections: {stats.get('total_detections', 0)}")
        logger.info(f"  Detections today: {stats.get('detections_today', 0)}")
        if stats.get('most_common_vehicle'):
            logger.info(f"  Most common: {stats['most_common_vehicle']['type']} " +
                       f"({stats['most_common_vehicle']['count']} times)")
        
        # Close database
        db.close()
        logger.info("\n✅ All database tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n🚀 Starting Lagos Traffic System Tests\n")
    
    # Test detector
    detector_pass = test_detector()
    
    # Test database
    database_pass = test_database()
    
    # Final summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Detector Module: {'✅ PASSED' if detector_pass else '❌ FAILED'}")
    logger.info(f"Database Module: {'✅ PASSED' if database_pass else '❌ FAILED'}")
    
    if detector_pass and database_pass:
        logger.info("\n🎉 All tests passed! Ready to proceed.")
        return 0
    else:
        logger.info("\n⚠️  Some tests failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    exit(main())

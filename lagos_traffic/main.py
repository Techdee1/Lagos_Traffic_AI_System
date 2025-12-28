"""
Lagos Traffic Analysis System - Main Orchestration Script
Coordinates video processing, vehicle detection, classification, and database logging
"""

import cv2
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
import signal
import sys

# Import modules
from modules.detector import LagosVehicleDetector
from modules.camera import VideoCamera, MultiVideoCamera
from modules.database import TrafficDatabase
from config import (
    TEST_VIDEOS_DIR, LOGS_DIR, DB_PATH,
    CONFIDENCE_THRESHOLD, STREAM_FPS
)

# Setup logging
LOGS_DIR.mkdir(exist_ok=True)
log_file = LOGS_DIR / f"traffic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LagosTrafficAnalyzer:
    """
    Main traffic analysis orchestrator
    """
    
    def __init__(self, video_source, confidence=None, loop=True):
        """
        Initialize traffic analyzer
        
        Args:
            video_source: Video file path, camera index, or list of videos
            confidence: Detection confidence threshold
            loop: Whether to loop videos
        """
        self.video_source = video_source
        self.loop = loop
        self.running = False
        
        # Initialize components
        logger.info("Initializing Lagos Traffic Analyzer...")
        
        # Detector
        logger.info("Loading vehicle detector...")
        self.detector = LagosVehicleDetector(confidence=confidence)
        
        # Database
        logger.info("Connecting to database...")
        self.db = TrafficDatabase(DB_PATH)
        
        # Camera
        logger.info("Setting up video source...")
        if isinstance(video_source, list):
            self.camera = MultiVideoCamera(
                video_source,
                loop_videos=loop,
                loop_sequence=loop
            )
        else:
            self.camera = VideoCamera(video_source, loop=loop)
        
        # Stats
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = None
        self.session_id = None
        
        # Vehicle counts (in-memory for quick access)
        self.vehicle_counts = {
            'okada': 0,
            'keke_napep': 0,
            'danfo': 0,
            'brt': 0,
            'private_car': 0,
            'truck': 0,
            'bus': 0
        }
        
        logger.info("Initialization complete")
    
    def process_frame(self, frame, frame_number):
        """
        Process a single frame: detect vehicles, classify, log to database
        
        Args:
            frame: OpenCV image
            frame_number: Frame number
        """
        try:
            # Run detection
            detections = self.detector.detect(frame)
            
            if detections:
                # Log to database
                self.db.log_detections_batch(detections, frame_number)
                
                # Update counts
                for det in detections:
                    vehicle_type = det['vehicle_type']
                    if vehicle_type in self.vehicle_counts:
                        self.vehicle_counts[vehicle_type] += 1
                    self.detection_count += len(detections)
                
                # Log summary
                if frame_number % 100 == 0:
                    logger.info(f"Frame {frame_number}: {len(detections)} vehicles detected")
            
            self.frame_count = frame_number
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {e}")
    
    def start(self):
        """Start traffic analysis"""
        if self.running:
            logger.warning("Analyzer already running")
            return
        
        self.running = True
        self.start_time = time.time()
        
        # Create session in database
        video_info = self.camera.get_info()
        self.session_id = self.db.create_session(str(video_info['source']))
        
        logger.info("=" * 60)
        logger.info("LAGOS TRAFFIC ANALYSIS STARTED")
        logger.info("=" * 60)
        logger.info(f"Video source: {video_info['source']}")
        logger.info(f"Resolution: {video_info['width']}x{video_info['height']}")
        logger.info(f"FPS: {video_info['fps']:.1f}")
        logger.info(f"Loop: {self.loop}")
        logger.info(f"Session ID: {self.session_id}")
        logger.info("=" * 60)
        
        # Start camera with frame callback
        self.camera.start(frame_callback=self.process_frame)
        
        logger.info("Traffic analysis running. Press Ctrl+C to stop.")
    
    def stop(self):
        """Stop traffic analysis"""
        if not self.running:
            return
        
        logger.info("Stopping traffic analysis...")
        self.running = False
        
        # Stop camera
        self.camera.stop()
        
        # Calculate stats
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Close session
        if self.session_id:
            self.db.close_session(self.session_id)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("TRAFFIC ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Duration: {elapsed_time:.1f} seconds")
        logger.info(f"Frames processed: {self.frame_count}")
        logger.info(f"Processing FPS: {fps:.1f}")
        logger.info(f"Total detections: {self.detection_count}")
        logger.info("")
        logger.info("Vehicle Counts:")
        for vehicle_type, count in sorted(self.vehicle_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                logger.info(f"  {vehicle_type}: {count}")
        logger.info("=" * 60)
    
    def run_live(self):
        """
        Run in live mode with visual display
        """
        self.running = True
        self.start_time = time.time()
        
        # Create session
        video_info = self.camera.get_info()
        self.session_id = self.db.create_session(str(video_info['source']))
        
        logger.info("Starting live display mode...")
        logger.info("Press 'q' to quit, 's' to show stats")
        
        cv2.namedWindow('Lagos Traffic Analysis', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Lagos Traffic Analysis', 1280, 720)
        
        try:
            while self.running:
                ret, frame = self.camera.read_frame()
                
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Run detection
                detections = self.detector.detect(frame)
                
                if detections:
                    # Log to database
                    self.db.log_detections_batch(detections, self.frame_count)
                    
                    # Update counts
                    for det in detections:
                        vehicle_type = det['vehicle_type']
                        if vehicle_type in self.vehicle_counts:
                            self.vehicle_counts[vehicle_type] += 1
                    
                    self.detection_count += len(detections)
                    
                    # Draw detections
                    frame = self.detector.draw_detections(frame, detections)
                
                # Draw info overlay
                self._draw_info_overlay(frame)
                
                # Display
                cv2.imshow('Lagos Traffic Analysis', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested by user")
                    break
                elif key == ord('s'):
                    self._print_stats()
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            cv2.destroyAllWindows()
            self.stop()
    
    def _draw_info_overlay(self, frame):
        """Draw information overlay on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Text info
        y_offset = 35
        line_height = 25
        
        cv2.putText(frame, "Lagos Traffic Analysis", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height + 10
        
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Detections: {self.detection_count}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += line_height
        
        # Vehicle counts
        cv2.putText(frame, "Vehicle Counts:", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += line_height
        
        for vehicle_type in ['okada', 'keke_napep', 'danfo', 'brt', 'private_car', 'truck']:
            count = self.vehicle_counts.get(vehicle_type, 0)
            if count > 0:
                cv2.putText(frame, f"  {vehicle_type}: {count}", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += line_height - 5
    
    def _print_stats(self):
        """Print current statistics"""
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "=" * 50)
        print("CURRENT STATISTICS")
        print("=" * 50)
        print(f"Runtime: {elapsed:.1f}s | Frames: {self.frame_count} | FPS: {fps:.1f}")
        print(f"Total detections: {self.detection_count}")
        print("\nVehicle Counts:")
        for vtype, count in sorted(self.vehicle_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  {vtype}: {count}")
        print("=" * 50 + "\n")
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        
        if self.camera:
            self.camera.release()
        
        if self.db:
            self.db.close()
        
        logger.info("Cleanup complete")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("\nReceived interrupt signal")
    sys.exit(0)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Lagos Traffic Analysis System'
    )
    parser.add_argument(
        'video',
        nargs='?',
        help='Video file path or camera index (default: test_videos/*.mp4)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f'Detection confidence threshold (default: {CONFIDENCE_THRESHOLD})'
    )
    parser.add_argument(
        '--no-loop',
        action='store_true',
        help='Disable video looping'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without visual display (background mode)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        help='Camera index (e.g., 0 for default camera)'
    )
    
    args = parser.parse_args()
    
    # Determine video source
    if args.camera is not None:
        video_source = args.camera
        logger.info(f"Using camera {args.camera}")
    elif args.video:
        video_source = args.video
        logger.info(f"Using video file: {args.video}")
    else:
        # Find all videos in test_videos directory
        video_files = list(TEST_VIDEOS_DIR.glob('*.mp4'))
        video_files.extend(TEST_VIDEOS_DIR.glob('*.avi'))
        
        if not video_files:
            logger.error(f"No video files found in {TEST_VIDEOS_DIR}")
            logger.info("Please provide a video file or add videos to test_videos/")
            return 1
        
        video_source = [str(v) for v in video_files]
        logger.info(f"Found {len(video_source)} video files")
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create analyzer
    try:
        analyzer = LagosTrafficAnalyzer(
            video_source,
            confidence=args.confidence,
            loop=not args.no_loop
        )
        
        if args.headless:
            # Run in background mode
            analyzer.start()
            
            # Keep running until interrupted
            try:
                while analyzer.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                analyzer.stop()
                analyzer.cleanup()
        else:
            # Run with visual display
            analyzer.run_live()
            analyzer.cleanup()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running traffic analyzer: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

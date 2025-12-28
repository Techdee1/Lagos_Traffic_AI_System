"""
Video Camera Handler Module
Manages video input, streaming, and frame processing
"""

import cv2
import threading
import time
import logging
from pathlib import Path
from typing import Optional, Callable

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import TEST_VIDEOS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoCamera:
    """
    Manages video input from files or camera streams
    Supports looping and frame callback processing
    """
    
    def __init__(self, video_source, loop=True):
        """
        Initialize video camera
        
        Args:
            video_source: Path to video file, camera index (0), or stream URL
            loop: Whether to loop video when it ends
        """
        self.video_source = video_source
        self.loop = loop
        self.cap = None
        self.running = False
        self.current_frame = None
        self.frame_count = 0
        self.fps = 30
        self.lock = threading.Lock()
        
        # Try to open video source
        self._open_video()
        
    def _open_video(self):
        """Open video source"""
        try:
            # Handle different source types
            if isinstance(self.video_source, int):
                # Camera index
                self.cap = cv2.VideoCapture(self.video_source)
                logger.info(f"Opened camera {self.video_source}")
            else:
                # File path or URL
                video_path = Path(self.video_source)
                
                # Check if file exists
                if not video_path.exists():
                    # Try in test_videos directory
                    video_path = TEST_VIDEOS_DIR / video_path.name
                    
                if video_path.exists():
                    self.cap = cv2.VideoCapture(str(video_path))
                    logger.info(f"Opened video file: {video_path}")
                else:
                    # Assume it's a URL
                    self.cap = cv2.VideoCapture(self.video_source)
                    logger.info(f"Opened video stream: {self.video_source}")
            
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open video source: {self.video_source}")
            
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video properties: {self.width}x{self.height} @ {self.fps:.1f} fps, "
                       f"{self.total_frames} frames")
            
        except Exception as e:
            logger.error(f"Error opening video source: {e}")
            raise
    
    def read_frame(self):
        """
        Read next frame from video
        
        Returns:
            tuple: (success, frame) or (False, None) if failed
        """
        if not self.cap or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            if self.loop and self.total_frames > 0:
                # Loop back to beginning
                logger.info("Video ended, looping back to start")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_count = 0
                ret, frame = self.cap.read()
            else:
                logger.info("Video ended")
                return False, None
        
        if ret:
            self.frame_count += 1
            with self.lock:
                self.current_frame = frame.copy()
        
        return ret, frame
    
    def get_current_frame(self):
        """
        Get the most recent frame (thread-safe)
        
        Returns:
            numpy.ndarray: Current frame or None
        """
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def start(self, frame_callback: Optional[Callable] = None):
        """
        Start video processing in a separate thread
        
        Args:
            frame_callback: Optional function to call for each frame
                           Signature: callback(frame, frame_count) -> None
        """
        if self.running:
            logger.warning("Camera already running")
            return
        
        self.running = True
        
        def process_video():
            logger.info("Video processing started")
            frame_delay = 1.0 / self.fps if self.fps > 0 else 0.033
            
            while self.running:
                start_time = time.time()
                
                ret, frame = self.read_frame()
                
                if not ret:
                    if not self.loop:
                        logger.info("Video ended, stopping")
                        self.running = False
                        break
                    continue
                
                # Call frame callback if provided
                if frame_callback:
                    try:
                        frame_callback(frame, self.frame_count)
                    except Exception as e:
                        logger.error(f"Error in frame callback: {e}")
                
                # Maintain frame rate
                elapsed = time.time() - start_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
            
            logger.info("Video processing stopped")
        
        # Start processing thread
        self.thread = threading.Thread(target=process_video, daemon=True)
        self.thread.start()
        logger.info("Video processing thread started")
    
    def stop(self):
        """Stop video processing"""
        logger.info("Stopping video processing...")
        self.running = False
        
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
        
        logger.info("Video processing stopped")
    
    def release(self):
        """Release video capture resources"""
        self.stop()
        
        if self.cap:
            self.cap.release()
            logger.info("Video capture released")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
    
    def get_frame_number(self):
        """Get current frame number"""
        return self.frame_count
    
    def get_progress(self):
        """
        Get video progress percentage
        
        Returns:
            float: Progress percentage (0-100) or 0 for live streams
        """
        if self.total_frames == 0:
            return 0.0
        
        return (self.frame_count / self.total_frames) * 100
    
    def get_info(self):
        """
        Get video information
        
        Returns:
            dict: Video properties
        """
        return {
            'source': str(self.video_source),
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'current_frame': self.frame_count,
            'progress': self.get_progress(),
            'running': self.running,
            'loop': self.loop
        }


class MultiVideoCamera:
    """
    Manages multiple video sources, switching between them sequentially
    """
    
    def __init__(self, video_sources, loop_videos=False, loop_sequence=True):
        """
        Initialize multi-video camera
        
        Args:
            video_sources: List of video file paths
            loop_videos: Whether to loop individual videos (False for multi-video mode)
            loop_sequence: Whether to loop the entire sequence
        """
        self.video_sources = video_sources
        self.loop_videos = loop_videos  # Should be False for proper video switching
        self.loop_sequence = loop_sequence
        self.current_index = 0
        self.current_camera = None
        
        if not video_sources:
            raise ValueError("No video sources provided")
        
        # Start with first video
        self._load_next_video()
    
    def _load_next_video(self):
        """Load the next video in sequence"""
        if self.current_camera:
            self.current_camera.release()
        
        video_source = self.video_sources[self.current_index]
        logger.info(f"Loading video {self.current_index + 1}/{len(self.video_sources)}: {video_source}")
        
        self.current_camera = VideoCamera(video_source, loop=self.loop_videos)
    
    def read_frame(self):
        """Read next frame, switching videos when needed"""
        ret, frame = self.current_camera.read_frame()
        
        if not ret:
            # Current video ended, move to next video
            logger.info(f"Video {self.current_index + 1} ended, switching to next video")
            self.current_index += 1
            
            if self.current_index >= len(self.video_sources):
                if self.loop_sequence:
                    # Loop back to first video
                    logger.info("All videos processed, looping back to first video")
                    self.current_index = 0
                else:
                    # No more videos
                    logger.info("All videos processed, ending")
                    return False, None
            
            # Load next video
            self._load_next_video()
            ret, frame = self.current_camera.read_frame()
        
        return ret, frame
    
    def get_current_frame(self):
        """Get current frame"""
        return self.current_camera.get_current_frame()
    
    def start(self, frame_callback: Optional[Callable] = None):
        """Start processing with automatic video switching"""
        
        def multi_video_callback(frame, frame_count):
            # Check if current video ended (not looping)
            if not self.loop_videos and frame_count >= self.current_camera.total_frames:
                self.current_index += 1
                
                if self.current_index < len(self.video_sources):
                    self._load_next_video()
                    self.current_camera.start(multi_video_callback)
                elif self.loop_sequence:
                    self.current_index = 0
                    self._load_next_video()
                    self.current_camera.start(multi_video_callback)
            
            # Call user callback
            if frame_callback:
                frame_callback(frame, frame_count)
        
        self.current_camera.start(multi_video_callback)
    
    def stop(self):
        """Stop processing"""
        if self.current_camera:
            self.current_camera.stop()
    
    def release(self):
        """Release all resources"""
        if self.current_camera:
            self.current_camera.release()
    
    def get_info(self):
        """Get current video info"""
        info = self.current_camera.get_info()
        info['video_index'] = self.current_index
        info['total_videos'] = len(self.video_sources)
        return info

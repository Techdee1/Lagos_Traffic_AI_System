"""
Traffic Database Module
Manages SQLite database for vehicle detection logging and analytics
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Optional

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficDatabase:
    """
    Manages vehicle detection storage and retrieval
    """
    
    def __init__(self, db_path=None):
        """Initialize database connection and create tables"""
        self.db_path = db_path or DB_PATH
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = None
        self.connect()
        self.create_tables()
        logger.info(f"Database initialized at {self.db_path}")
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self.conn.row_factory = sqlite3.Row
            logger.info("Database connection established")
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Vehicle detections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                vehicle_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox TEXT NOT NULL,
                frame_id INTEGER,
                class_id INTEGER,
                class_name TEXT
            )
        """)
        
        # Vehicle counts summary table (per session/time window)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                vehicle_type TEXT NOT NULL,
                count INTEGER NOT NULL,
                time_window INTEGER DEFAULT 60
            )
        """)
        
        # Session tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                video_source TEXT,
                total_detections INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active'
            )
        """)
        
        # Create indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_detections_timestamp 
            ON vehicle_detections(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_detections_vehicle_type 
            ON vehicle_detections(vehicle_type)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_counts_timestamp 
            ON vehicle_counts(timestamp)
        """)
        
        self.conn.commit()
        logger.info("Database tables created/verified")
    
    def log_detection(self, detection: Dict):
        """
        Log a single vehicle detection
        
        Args:
            detection: Dict with keys: vehicle_type, confidence, bbox, 
                      frame_id, class_id, class_name
        """
        try:
            cursor = self.conn.cursor()
            
            # Convert bbox list to JSON string
            bbox_json = json.dumps(detection['bbox'])
            
            cursor.execute("""
                INSERT INTO vehicle_detections 
                (vehicle_type, confidence, bbox, frame_id, class_id, class_name)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                detection['vehicle_type'],
                detection['confidence'],
                bbox_json,
                detection.get('frame_id', 0),
                detection.get('class_id'),
                detection.get('class_name')
            ))
            
            self.conn.commit()
            return cursor.lastrowid
            
        except sqlite3.Error as e:
            logger.error(f"Error logging detection: {e}")
            return None
    
    def log_detections_batch(self, detections: List[Dict], frame_id: int = 0):
        """
        Log multiple detections in batch
        
        Args:
            detections: List of detection dicts
            frame_id: Current frame number
        """
        try:
            cursor = self.conn.cursor()
            
            data = [
                (
                    det['vehicle_type'],
                    det['confidence'],
                    json.dumps(det['bbox']),
                    frame_id,
                    det.get('class_id'),
                    det.get('class_name')
                )
                for det in detections
            ]
            
            cursor.executemany("""
                INSERT INTO vehicle_detections 
                (vehicle_type, confidence, bbox, frame_id, class_id, class_name)
                VALUES (?, ?, ?, ?, ?, ?)
            """, data)
            
            self.conn.commit()
            logger.debug(f"Logged {len(detections)} detections for frame {frame_id}")
            
        except sqlite3.Error as e:
            logger.error(f"Error logging batch detections: {e}")
    
    def get_recent_detections(self, limit: int = 50) -> List[Dict]:
        """
        Get most recent vehicle detections
        
        Args:
            limit: Maximum number of detections to return
            
        Returns:
            List of detection dicts
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT id, timestamp, vehicle_type, confidence, bbox, 
                       frame_id, class_id, class_name
                FROM vehicle_detections
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            
            detections = []
            for row in rows:
                detections.append({
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'vehicle_type': row['vehicle_type'],
                    'confidence': row['confidence'],
                    'bbox': json.loads(row['bbox']),
                    'frame_id': row['frame_id'],
                    'class_id': row['class_id'],
                    'class_name': row['class_name']
                })
            
            return detections
            
        except sqlite3.Error as e:
            logger.error(f"Error fetching recent detections: {e}")
            return []
    
    def get_vehicle_counts(self, time_window: int = 60) -> Dict[str, int]:
        """
        Get vehicle counts for the last N seconds
        
        Args:
            time_window: Time window in seconds
            
        Returns:
            Dict mapping vehicle_type to count
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT vehicle_type, COUNT(*) as count
                FROM vehicle_detections
                WHERE timestamp >= datetime('now', '-' || ? || ' seconds')
                GROUP BY vehicle_type
                ORDER BY count DESC
            """, (time_window,))
            
            rows = cursor.fetchall()
            
            counts = {row['vehicle_type']: row['count'] for row in rows}
            
            return counts
            
        except sqlite3.Error as e:
            logger.error(f"Error fetching vehicle counts: {e}")
            return {}
    
    def get_total_counts(self) -> Dict[str, int]:
        """
        Get total vehicle counts (all time)
        
        Returns:
            Dict mapping vehicle_type to total count
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT vehicle_type, COUNT(*) as count
                FROM vehicle_detections
                GROUP BY vehicle_type
                ORDER BY count DESC
            """)
            
            rows = cursor.fetchall()
            
            counts = {row['vehicle_type']: row['count'] for row in rows}
            
            return counts
            
        except sqlite3.Error as e:
            logger.error(f"Error fetching total counts: {e}")
            return {}
    
    def get_counts_by_hour(self, hours: int = 24) -> List[Dict]:
        """
        Get vehicle counts grouped by hour
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of dicts with hour, vehicle_type, count
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT 
                    strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                    vehicle_type,
                    COUNT(*) as count
                FROM vehicle_detections
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
                GROUP BY hour, vehicle_type
                ORDER BY hour DESC, count DESC
            """, (hours,))
            
            rows = cursor.fetchall()
            
            results = [
                {
                    'hour': row['hour'],
                    'vehicle_type': row['vehicle_type'],
                    'count': row['count']
                }
                for row in rows
            ]
            
            return results
            
        except sqlite3.Error as e:
            logger.error(f"Error fetching hourly counts: {e}")
            return []
    
    def create_session(self, video_source: str) -> int:
        """
        Create a new monitoring session
        
        Args:
            video_source: Path or name of video source
            
        Returns:
            Session ID
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO sessions (video_source, status)
                VALUES (?, 'active')
            """, (video_source,))
            
            self.conn.commit()
            session_id = cursor.lastrowid
            logger.info(f"Created session {session_id} for {video_source}")
            
            return session_id
            
        except sqlite3.Error as e:
            logger.error(f"Error creating session: {e}")
            return None
    
    def close_session(self, session_id: int):
        """
        Close a monitoring session
        
        Args:
            session_id: Session ID to close
        """
        try:
            cursor = self.conn.cursor()
            
            # Get total detections for this session
            cursor.execute("""
                SELECT COUNT(*) as total
                FROM vehicle_detections
                WHERE timestamp >= (
                    SELECT start_time FROM sessions WHERE id = ?
                )
            """, (session_id,))
            
            row = cursor.fetchone()
            total = row['total'] if row else 0
            
            # Update session
            cursor.execute("""
                UPDATE sessions
                SET end_time = CURRENT_TIMESTAMP,
                    status = 'completed',
                    total_detections = ?
                WHERE id = ?
            """, (total, session_id))
            
            self.conn.commit()
            logger.info(f"Closed session {session_id} with {total} detections")
            
        except sqlite3.Error as e:
            logger.error(f"Error closing session: {e}")
    
    def clear_old_data(self, days: int = 30):
        """
        Delete detections older than N days
        
        Args:
            days: Number of days to keep
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                DELETE FROM vehicle_detections
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days,))
            
            deleted = cursor.rowcount
            self.conn.commit()
            
            logger.info(f"Deleted {deleted} old detection records")
            
        except sqlite3.Error as e:
            logger.error(f"Error clearing old data: {e}")
    
    def get_stats(self) -> Dict:
        """
        Get database statistics
        
        Returns:
            Dict with various statistics
        """
        try:
            cursor = self.conn.cursor()
            
            # Total detections
            cursor.execute("SELECT COUNT(*) as total FROM vehicle_detections")
            total = cursor.fetchone()['total']
            
            # Detections today
            cursor.execute("""
                SELECT COUNT(*) as today
                FROM vehicle_detections
                WHERE date(timestamp) = date('now')
            """)
            today = cursor.fetchone()['today']
            
            # Most common vehicle
            cursor.execute("""
                SELECT vehicle_type, COUNT(*) as count
                FROM vehicle_detections
                GROUP BY vehicle_type
                ORDER BY count DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            most_common = {
                'type': row['vehicle_type'],
                'count': row['count']
            } if row else None
            
            return {
                'total_detections': total,
                'detections_today': today,
                'most_common_vehicle': most_common,
                'database_path': str(self.db_path)
            }
            
        except sqlite3.Error as e:
            logger.error(f"Error fetching stats: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

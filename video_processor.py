import cv2
import numpy as np
import threading
import time
from datetime import datetime
from ultralytics import YOLO
import mediapipe as mp
from collections import deque
import json

class VideoProcessor:
    def __init__(self, video_path=None, camera_index=None, session_id=None, socketio=None):
        self.video_path = video_path
        self.camera_index = camera_index
        self.session_id = session_id
        self.socketio = socketio
        self.created_at = datetime.now()
        
        # Processing state
        self.is_processing = False
        self.is_live = camera_index is not None
        self.cap = None
        
        # Tracking data storage
        self.tracking_data = {
            'players': [],
            'ball': [],
            'frames': [],
            'timestamps': []
        }
        
        # Initialize models
        self.initialize_models()
        
        # Performance metrics
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
    def initialize_models(self):
        """Initialize YOLO and MediaPipe models"""
        try:
            # Load YOLO model for object detection
            self.yolo_model = YOLO('yolov8n.pt')
            
            # Initialize MediaPipe for pose estimation
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            print("Models initialized successfully")
            
        except Exception as e:
            print(f"Error initializing models: {e}")
            # Fallback to basic OpenCV detection
            self.yolo_model = None
            self.pose = None
    
    def start_live_processing(self, client_id):
        """Start live video processing"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            
            self.is_processing = True
            print(f"Started live processing for session {self.session_id}")
            
            while self.is_processing:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, analysis_data = self.process_frame(frame)
                
                # Send data to client
                if self.socketio:
                    self.socketio.emit('frame_data', {
                        'session_id': self.session_id,
                        'frame': self.frame_to_base64(processed_frame),
                        'analysis': analysis_data
                    }, room=client_id)
                
                # Control frame rate
                time.sleep(1/30)  # 30 FPS
                
        except Exception as e:
            print(f"Error in live processing: {e}")
            if self.socketio:
                self.socketio.emit('error', {'message': str(e)}, room=client_id)
        finally:
            self.stop()
    
    def start_upload_processing(self, client_id):
        """Start processing uploaded video"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise Exception("Could not open video file")
            
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.is_processing = True
            print(f"Started upload processing for session {self.session_id}")
            
            frame_count = 0
            while self.is_processing:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, analysis_data = self.process_frame(frame)
                
                # Update progress
                frame_count += 1
                progress = (frame_count / total_frames) * 100
                
                # Send data to client
                if self.socketio:
                    self.socketio.emit('frame_data', {
                        'session_id': self.session_id,
                        'frame': self.frame_to_base64(processed_frame),
                        'analysis': analysis_data,
                        'progress': progress,
                        'current_frame': frame_count,
                        'total_frames': total_frames
                    }, room=client_id)
                
                # Control processing speed
                time.sleep(1/fps)
                
        except Exception as e:
            print(f"Error in upload processing: {e}")
            if self.socketio:
                self.socketio.emit('error', {'message': str(e)}, room=client_id)
        finally:
            self.stop()
    
    def process_frame(self, frame):
        """Process a single frame and return analysis data"""
        try:
            # Store original frame dimensions
            height, width = frame.shape[:2]
            
            # Detect players and ball
            player_detections = self.detect_players(frame)
            ball_detection = self.detect_ball(frame)
            
            # Track objects
            tracked_players = self.track_players(player_detections, frame)
            tracked_ball = self.track_ball(ball_detection, frame)
            
            # Draw overlays
            processed_frame = self.draw_overlays(frame, tracked_players, tracked_ball)
            
            # Store tracking data
            frame_data = {
                'timestamp': time.time(),
                'players': tracked_players,
                'ball': tracked_ball,
                'frame_number': len(self.tracking_data['frames'])
            }
            
            self.tracking_data['frames'].append(frame_data)
            self.tracking_data['players'].append(tracked_players)
            self.tracking_data['ball'].append(tracked_ball)
            self.tracking_data['timestamps'].append(frame_data['timestamp'])
            
            # Calculate FPS
            self.update_fps()
            
            # Prepare analysis data
            analysis_data = {
                'fps': self.current_fps,
                'player_count': len(tracked_players),
                'ball_detected': ball_detection is not None,
                'frame_number': frame_data['frame_number']
            }
            
            return processed_frame, analysis_data
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame, {}
    
    def detect_players(self, frame):
        """Detect players in the frame using YOLO"""
        try:
            if self.yolo_model is None:
                return []
            
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False)
            
            player_detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Filter for person class (class 0 in COCO dataset)
                        if int(box.cls[0]) == 0:  # Person class
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0])
                            
                            if confidence > 0.5:  # Confidence threshold
                                player_detections.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': confidence,
                                    'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                                })
            
            return player_detections
            
        except Exception as e:
            print(f"Error detecting players: {e}")
            return []
    
    def detect_ball(self, frame):
        """Detect ball in the frame"""
        try:
            # Convert to HSV for better ball detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define ball color ranges (adjust based on sport)
            # For soccer ball (white/black)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            
            # Create mask
            mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (likely the ball)
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 100:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    return {
                        'bbox': [x, y, x + w, y + h],
                        'center': [x + w//2, y + h//2],
                        'area': area
                    }
            
            return None
            
        except Exception as e:
            print(f"Error detecting ball: {e}")
            return None
    
    def track_players(self, detections, frame):
        """Track players across frames"""
        tracked_players = []
        
        for i, detection in enumerate(detections):
            # Simple tracking: assign ID based on position
            # In a real implementation, you'd use more sophisticated tracking
            player_id = f"player_{i}"
            
            tracked_players.append({
                'id': player_id,
                'bbox': detection['bbox'],
                'center': detection['center'],
                'confidence': detection['confidence']
            })
        
        return tracked_players
    
    def track_ball(self, detection, frame):
        """Track ball across frames"""
        if detection is None:
            return None
        
        return {
            'bbox': detection['bbox'],
            'center': detection['center'],
            'area': detection['area']
        }
    
    def draw_overlays(self, frame, players, ball):
        """Draw tracking overlays on frame"""
        # Draw player bounding boxes and IDs
        for player in players:
            x1, y1, x2, y2 = player['bbox']
            center = player['center']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw player ID
            cv2.putText(frame, player['id'], (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(frame, tuple(center), 3, (255, 0, 0), -1)
        
        # Draw ball
        if ball:
            x1, y1, x2, y2 = ball['bbox']
            center = ball['center']
            
            # Draw ball bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw ball center
            cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def frame_to_base64(self, frame):
        """Convert frame to base64 for transmission"""
        try:
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            # Convert to base64
            import base64
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Error converting frame to base64: {e}")
            return ""
    
    def get_tracking_data(self):
        """Get all tracking data"""
        return self.tracking_data
    
    def get_analysis_results(self):
        """Get comprehensive analysis results"""
        return {
            'session_id': self.session_id,
            'total_frames': len(self.tracking_data['frames']),
            'duration': self.tracking_data['timestamps'][-1] - self.tracking_data['timestamps'][0] if self.tracking_data['timestamps'] else 0,
            'average_fps': self.current_fps,
            'player_detections': len([f for f in self.tracking_data['frames'] if f['players']]),
            'ball_detections': len([f for f in self.tracking_data['frames'] if f['ball']])
        }
    
    def get_status(self):
        """Get current processing status"""
        return {
            'is_processing': self.is_processing,
            'is_live': self.is_live,
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat()
        }
    
    def stop(self):
        """Stop video processing"""
        self.is_processing = False
        if self.cap:
            self.cap.release()
        print(f"Stopped processing for session {self.session_id}") 
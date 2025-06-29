import cv2
import numpy as np
from collections import deque
import time

class BallTracker:
    def __init__(self):
        self.ball_history = deque(maxlen=50)  # Store last 50 ball positions
        self.last_ball_position = None
        self.ball_velocity = [0, 0]
        self.ball_acceleration = [0, 0]
        
        # Ball detection parameters
        self.min_ball_area = 50
        self.max_ball_area = 5000
        self.ball_color_ranges = [
            # White ball (soccer)
            {'lower': np.array([0, 0, 200]), 'upper': np.array([180, 30, 255])},
            # Orange ball (basketball)
            {'lower': np.array([5, 50, 50]), 'upper': np.array([15, 255, 255])},
            # Brown ball (football)
            {'lower': np.array([10, 50, 20]), 'upper': np.array([20, 255, 200])}
        ]
        
    def detect_ball(self, frame):
        """Detect ball in the frame using multiple color ranges"""
        best_detection = None
        best_confidence = 0
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for color_range in self.ball_color_ranges:
            # Create mask for this color range
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if self.min_ball_area < area < self.max_ball_area:
                    # Calculate circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # Higher circularity indicates more ball-like shape
                        if circularity > 0.6:  # Circularity threshold
                            x, y, w, h = cv2.boundingRect(contour)
                            center = (x + w//2, y + h//2)
                            
                            # Calculate confidence based on circularity and area
                            confidence = circularity * (area / self.max_ball_area)
                            
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_detection = {
                                    'bbox': [x, y, x + w, y + h],
                                    'center': center,
                                    'area': area,
                                    'circularity': circularity,
                                    'confidence': confidence
                                }
        
        return best_detection if best_confidence > 0.3 else None
    
    def track_ball(self, detection, frame):
        """Track ball across frames and calculate velocity"""
        if detection is None:
            # No ball detected, update history with None
            self.ball_history.append(None)
            return None
        
        current_position = detection['center']
        current_time = time.time()
        
        # Update ball history
        self.ball_history.append({
            'position': current_position,
            'time': current_time,
            'bbox': detection['bbox'],
            'area': detection['area'],
            'confidence': detection['confidence']
        })
        
        # Calculate velocity if we have previous position
        if self.last_ball_position is not None:
            prev_pos, prev_time = self.last_ball_position
            
            # Calculate velocity (pixels per second)
            dt = current_time - prev_time
            if dt > 0:
                dx = current_position[0] - prev_pos[0]
                dy = current_position[1] - prev_pos[1]
                
                self.ball_velocity = [dx / dt, dy / dt]
                
                # Calculate speed
                speed = np.sqrt(dx*dx + dy*dy) / dt
                detection['speed'] = speed
                detection['velocity'] = self.ball_velocity
        
        # Update last position
        self.last_ball_position = (current_position, current_time)
        
        return detection
    
    def predict_ball_position(self, frames_ahead=5):
        """Predict ball position based on current velocity"""
        if len(self.ball_history) < 2 or self.last_ball_position is None:
            return None
        
        current_pos, current_time = self.last_ball_position
        
        # Simple linear prediction
        predicted_x = current_pos[0] + self.ball_velocity[0] * frames_ahead / 30  # Assuming 30 FPS
        predicted_y = current_pos[1] + self.ball_velocity[1] * frames_ahead / 30
        
        return (int(predicted_x), int(predicted_y))
    
    def analyze_ball_movement(self):
        """Analyze ball movement patterns"""
        if len(self.ball_history) < 10:
            return None
        
        # Filter out None values
        valid_positions = [entry for entry in self.ball_history if entry is not None]
        
        if len(valid_positions) < 5:
            return None
        
        # Calculate movement statistics
        speeds = []
        directions = []
        
        for i in range(1, len(valid_positions)):
            prev_pos = valid_positions[i-1]['position']
            curr_pos = valid_positions[i]['position']
            dt = valid_positions[i]['time'] - valid_positions[i-1]['time']
            
            if dt > 0:
                dx = curr_pos[0] - prev_pos[0]
                dy = curr_pos[1] - prev_pos[1]
                
                speed = np.sqrt(dx*dx + dy*dy) / dt
                speeds.append(speed)
                
                direction = np.arctan2(dy, dx)
                directions.append(direction)
        
        if not speeds:
            return None
        
        return {
            'average_speed': np.mean(speeds),
            'max_speed': np.max(speeds),
            'speed_variance': np.var(speeds),
            'movement_direction': np.mean(directions),
            'total_distance': sum(speeds) / 30,  # Approximate total distance
            'ball_possession_time': len(valid_positions) / 30  # Seconds
        }
    
    def detect_ball_events(self, frame_width, frame_height):
        """Detect ball events like shots, passes, etc."""
        if len(self.ball_history) < 5:
            return []
        
        events = []
        valid_positions = [entry for entry in self.ball_history if entry is not None]
        
        if len(valid_positions) < 3:
            return events
        
        # Detect high-speed events (shots, passes)
        for i in range(1, len(valid_positions)):
            prev_pos = valid_positions[i-1]['position']
            curr_pos = valid_positions[i]['position']
            dt = valid_positions[i]['time'] - valid_positions[i-1]['time']
            
            if dt > 0:
                dx = curr_pos[0] - prev_pos[0]
                dy = curr_pos[1] - prev_pos[1]
                speed = np.sqrt(dx*dx + dy*dy) / dt
                
                # High speed event detection
                if speed > 200:  # Threshold for high-speed events
                    event_type = "shot" if speed > 300 else "pass"
                    events.append({
                        'type': event_type,
                        'speed': speed,
                        'position': curr_pos,
                        'time': valid_positions[i]['time']
                    })
        
        # Detect ball out of bounds
        if valid_positions:
            last_pos = valid_positions[-1]['position']
            if (last_pos[0] < 0 or last_pos[0] > frame_width or 
                last_pos[1] < 0 or last_pos[1] > frame_height):
                events.append({
                    'type': 'out_of_bounds',
                    'position': last_pos,
                    'time': valid_positions[-1]['time']
                })
        
        return events
    
    def get_ball_trajectory(self):
        """Get the ball trajectory"""
        trajectory = []
        for entry in self.ball_history:
            if entry is not None:
                trajectory.append({
                    'position': entry['position'],
                    'time': entry['time'],
                    'area': entry['area'],
                    'confidence': entry['confidence']
                })
        return trajectory
    
    def calculate_possession_metrics(self, player_positions):
        """Calculate ball possession metrics for players"""
        if not player_positions or not self.ball_history:
            return {}
        
        # Find the last valid ball position
        last_ball_entry = None
        for entry in reversed(self.ball_history):
            if entry is not None:
                last_ball_entry = entry
                break
        
        if last_ball_entry is None:
            return {}
        
        ball_pos = last_ball_entry['position']
        
        # Find closest player to ball
        closest_player = None
        min_distance = float('inf')
        
        for player in player_positions:
            player_center = player['center']
            distance = np.sqrt(
                (ball_pos[0] - player_center[0])**2 + 
                (ball_pos[1] - player_center[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_player = player
        
        if closest_player and min_distance < 50:  # Ball possession threshold
            return {
                'player_id': closest_player['id'],
                'distance_to_ball': min_distance,
                'ball_confidence': last_ball_entry['confidence']
            }
        
        return {}
    
    def reset(self):
        """Reset ball tracking data"""
        self.ball_history.clear()
        self.last_ball_position = None
        self.ball_velocity = [0, 0]
        self.ball_acceleration = [0, 0] 
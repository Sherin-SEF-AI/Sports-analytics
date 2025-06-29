import cv2
import numpy as np
from collections import deque
import time

class PlayerTracker:
    def __init__(self):
        self.tracked_players = {}
        self.next_player_id = 0
        self.max_disappeared = 30  # Frames before considering player lost
        self.max_distance = 100  # Maximum distance for tracking
        
        # Player trajectory storage
        self.trajectories = {}
        
    def update_tracking(self, detections, frame):
        """Update player tracking with new detections"""
        if not detections:
            # No detections, mark all players as disappeared
            for player_id in list(self.tracked_players.keys()):
                self.tracked_players[player_id]['disappeared'] += 1
                if self.tracked_players[player_id]['disappeared'] > self.max_disappeared:
                    del self.tracked_players[player_id]
            return []
        
        # If no tracked players, initialize with current detections
        if not self.tracked_players:
            for detection in detections:
                self.register_player(detection)
            return self.get_tracked_players()
        
        # Calculate distances between tracked players and new detections
        tracked_centers = [player['center'] for player in self.tracked_players.values()]
        detection_centers = [det['center'] for det in detections]
        
        # Simple distance-based matching
        matched_detections = set()
        matched_trackers = set()
        
        for i, tracked_center in enumerate(tracked_centers):
            min_distance = float('inf')
            best_match = None
            
            for j, detection_center in enumerate(detection_centers):
                if j in matched_detections:
                    continue
                
                distance = np.sqrt(
                    (tracked_center[0] - detection_center[0])**2 + 
                    (tracked_center[1] - detection_center[1])**2
                )
                
                if distance < min_distance and distance < self.max_distance:
                    min_distance = distance
                    best_match = j
            
            if best_match is not None:
                player_id = list(self.tracked_players.keys())[i]
                self.update_player(player_id, detections[best_match])
                matched_detections.add(best_match)
                matched_trackers.add(i)
        
        # Handle unmatched detections (new players)
        for i, detection in enumerate(detections):
            if i not in matched_detections:
                self.register_player(detection)
        
        # Handle unmatched trackers (disappeared players)
        for i, player_id in enumerate(list(self.tracked_players.keys())):
            if i not in matched_trackers:
                self.tracked_players[player_id]['disappeared'] += 1
                if self.tracked_players[player_id]['disappeared'] > self.max_disappeared:
                    del self.tracked_players[player_id]
        
        return self.get_tracked_players()
    
    def register_player(self, detection):
        """Register a new player"""
        player_id = f"player_{self.next_player_id}"
        self.next_player_id += 1
        
        self.tracked_players[player_id] = {
            'id': player_id,
            'bbox': detection['bbox'],
            'center': detection['center'],
            'confidence': detection['confidence'],
            'disappeared': 0,
            'first_seen': time.time(),
            'last_seen': time.time()
        }
        
        # Initialize trajectory
        self.trajectories[player_id] = deque(maxlen=100)
        self.trajectories[player_id].append(detection['center'])
    
    def update_player(self, player_id, detection):
        """Update existing player with new detection"""
        if player_id in self.tracked_players:
            self.tracked_players[player_id].update({
                'bbox': detection['bbox'],
                'center': detection['center'],
                'confidence': detection['confidence'],
                'disappeared': 0,
                'last_seen': time.time()
            })
            
            # Update trajectory
            if player_id in self.trajectories:
                self.trajectories[player_id].append(detection['center'])
    
    def get_tracked_players(self):
        """Get current tracked players"""
        return list(self.tracked_players.values())
    
    def get_player_trajectory(self, player_id):
        """Get trajectory for a specific player"""
        return list(self.trajectories.get(player_id, []))
    
    def get_all_trajectories(self):
        """Get all player trajectories"""
        return {player_id: list(trajectory) for player_id, trajectory in self.trajectories.items()}
    
    def calculate_player_metrics(self, player_id):
        """Calculate performance metrics for a player"""
        if player_id not in self.tracked_players or player_id not in self.trajectories:
            return None
        
        trajectory = list(self.trajectories[player_id])
        if len(trajectory) < 2:
            return None
        
        # Calculate total distance
        total_distance = 0
        speeds = []
        
        for i in range(1, len(trajectory)):
            prev_point = trajectory[i-1]
            curr_point = trajectory[i]
            
            distance = np.sqrt(
                (curr_point[0] - prev_point[0])**2 + 
                (curr_point[1] - prev_point[1])**2
            )
            total_distance += distance
            
            # Calculate speed (assuming 30 FPS)
            speed = distance * 30  # pixels per second
            speeds.append(speed)
        
        # Calculate metrics
        avg_speed = np.mean(speeds) if speeds else 0
        max_speed = np.max(speeds) if speeds else 0
        min_speed = np.min(speeds) if speeds else 0
        
        # Count sprints (speed > threshold)
        sprint_threshold = np.mean(speeds) + np.std(speeds) if speeds else 0
        sprint_count = sum(1 for speed in speeds if speed > sprint_threshold)
        
        return {
            'player_id': player_id,
            'total_distance': total_distance,
            'average_speed': avg_speed,
            'max_speed': max_speed,
            'min_speed': min_speed,
            'sprint_count': sprint_count,
            'tracking_duration': time.time() - self.tracked_players[player_id]['first_seen']
        }
    
    def get_team_formation(self, frame_width, frame_height):
        """Analyze team formation based on player positions"""
        if not self.tracked_players:
            return None
        
        # Get current player positions
        positions = []
        for player in self.tracked_players.values():
            if player['disappeared'] == 0:  # Only active players
                x, y = player['center']
                # Normalize positions to 0-1 range
                norm_x = x / frame_width
                norm_y = y / frame_height
                positions.append((norm_x, norm_y))
        
        if len(positions) < 4:  # Need at least 4 players for formation analysis
            return None
        
        # Simple formation detection based on player distribution
        # This is a basic implementation - more sophisticated algorithms would be used in production
        
        # Sort players by Y position (field position)
        positions.sort(key=lambda p: p[1])
        
        # Count players in different field zones
        defenders = sum(1 for x, y in positions if y < 0.33)
        midfielders = sum(1 for x, y in positions if 0.33 <= y <= 0.67)
        forwards = sum(1 for x, y in positions if y > 0.67)
        
        # Determine formation
        if defenders == 4 and midfielders == 4 and forwards == 2:
            formation = "4-4-2"
        elif defenders == 4 and midfielders == 3 and forwards == 3:
            formation = "4-3-3"
        elif defenders == 3 and midfielders == 5 and forwards == 2:
            formation = "3-5-2"
        else:
            formation = f"{defenders}-{midfielders}-{forwards}"
        
        return {
            'formation': formation,
            'defenders': defenders,
            'midfielders': midfielders,
            'forwards': forwards,
            'total_players': len(positions)
        }
    
    def reset(self):
        """Reset all tracking data"""
        self.tracked_players = {}
        self.trajectories = {}
        self.next_player_id = 0 
import numpy as np
import time
from collections import deque

class RefereeAssistant:
    def __init__(self):
        self.violations = []
        self.offside_history = deque(maxlen=100)
        self.foul_detections = deque(maxlen=50)
        self.field_boundaries = None
        
    def detect_offside(self, player_positions, ball_position, frame_width, frame_height):
        """Detect offside situations in football"""
        if not player_positions or not ball_position:
            return None
        
        # Define field zones (simplified)
        # In a real implementation, this would use proper field calibration
        midfield_line = frame_width / 2
        
        # Find the ball's X position
        ball_x = ball_position[0]
        
        # Determine which half the ball is in
        attacking_half = ball_x > midfield_line
        
        offside_players = []
        
        for player in player_positions:
            player_x = player['center'][0]
            
            # Check if player is in attacking half
            if attacking_half and player_x > midfield_line:
                # Check if player is beyond the second-to-last defender
                defenders_in_attacking_half = [
                    p for p in player_positions 
                    if p['center'][0] > midfield_line and p['id'] != player['id']
                ]
                
                if len(defenders_in_attacking_half) < 2:
                    # Player is offside
                    offside_players.append({
                        'player_id': player['id'],
                        'position': player['center'],
                        'severity': 'offside',
                        'timestamp': time.time()
                    })
        
        if offside_players:
            self.offside_history.append({
                'players': offside_players,
                'ball_position': ball_position,
                'timestamp': time.time()
            })
        
        return offside_players
    
    def detect_fouls(self, player_positions, ball_position, frame_width, frame_height):
        """Detect potential fouls based on player interactions"""
        if len(player_positions) < 2:
            return []
        
        fouls = []
        
        # Check for player collisions (potential fouls)
        for i, player1 in enumerate(player_positions):
            for j, player2 in enumerate(player_positions[i+1:], i+1):
                pos1 = player1['center']
                pos2 = player2['center']
                
                # Calculate distance between players
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                # If players are very close, potential foul
                if distance < 30:  # Threshold for close contact
                    fouls.append({
                        'type': 'potential_foul',
                        'player1': player1['id'],
                        'player2': player2['id'],
                        'distance': distance,
                        'position': pos1,
                        'severity': 'medium',
                        'timestamp': time.time()
                    })
        
        # Check for ball-related fouls
        if ball_position:
            for player in player_positions:
                player_pos = player['center']
                ball_distance = np.sqrt(
                    (player_pos[0] - ball_position[0])**2 + 
                    (player_pos[1] - ball_position[1])**2
                )
                
                # If player is very close to ball, check for handball
                if ball_distance < 20:  # Very close to ball
                    # This would require more sophisticated analysis in real implementation
                    pass
        
        if fouls:
            self.foul_detections.extend(fouls)
        
        return fouls
    
    def detect_out_of_bounds(self, ball_position, frame_width, frame_height):
        """Detect if ball is out of bounds"""
        if not ball_position:
            return None
        
        x, y = ball_position
        
        # Check if ball is outside the frame
        if x < 0 or x > frame_width or y < 0 or y > frame_height:
            return {
                'type': 'out_of_bounds',
                'ball_position': ball_position,
                'timestamp': time.time()
            }
        
        return None
    
    def detect_goal_events(self, ball_position, frame_width, frame_height):
        """Detect potential goal events"""
        if not ball_position:
            return None
        
        x, y = ball_position
        
        # Define goal areas (simplified)
        # In real implementation, this would use proper goal detection
        goal_width = 100
        goal_height = 50
        
        # Left goal (x = 0)
        if x < goal_width and frame_height/2 - goal_height/2 < y < frame_height/2 + goal_height/2:
            return {
                'type': 'goal',
                'team': 'away',  # Assuming left side is away team
                'ball_position': ball_position,
                'timestamp': time.time()
            }
        
        # Right goal (x = frame_width)
        if x > frame_width - goal_width and frame_height/2 - goal_height/2 < y < frame_height/2 + goal_height/2:
            return {
                'type': 'goal',
                'team': 'home',  # Assuming right side is home team
                'ball_position': ball_position,
                'timestamp': time.time()
            }
        
        return None
    
    def analyze_referee_decisions(self, tracking_data):
        """Analyze all referee-related decisions from tracking data"""
        if not tracking_data or not tracking_data.get('frames'):
            return None
        
        latest_frame = tracking_data['frames'][-1]
        player_positions = latest_frame.get('players', [])
        ball_position = latest_frame.get('ball', {}).get('center') if latest_frame.get('ball') else None
        
        # Get frame dimensions (default if not available)
        frame_width = 1920
        frame_height = 1080
        
        # Run all detection algorithms
        offside_detections = self.detect_offside(player_positions, ball_position, frame_width, frame_height)
        foul_detections = self.detect_fouls(player_positions, ball_position, frame_width, frame_height)
        out_of_bounds = self.detect_out_of_bounds(ball_position, frame_width, frame_height)
        goal_events = self.detect_goal_events(ball_position, frame_width, frame_height)
        
        # Compile all decisions
        decisions = {
            'offside': offside_detections,
            'fouls': foul_detections,
            'out_of_bounds': out_of_bounds,
            'goal_events': goal_events,
            'timestamp': time.time()
        }
        
        # Store violations
        if any([offside_detections, foul_detections, out_of_bounds, goal_events]):
            self.violations.append(decisions)
        
        return decisions
    
    def get_referee_statistics(self):
        """Get statistics about referee decisions"""
        if not self.violations:
            return None
        
        total_violations = len(self.violations)
        offside_count = sum(1 for v in self.violations if v.get('offside'))
        foul_count = sum(1 for v in self.violations if v.get('fouls'))
        out_of_bounds_count = sum(1 for v in self.violations if v.get('out_of_bounds'))
        goal_count = sum(1 for v in self.violations if v.get('goal_events'))
        
        return {
            'total_violations': total_violations,
            'offside_count': offside_count,
            'foul_count': foul_count,
            'out_of_bounds_count': out_of_bounds_count,
            'goal_count': goal_count,
            'violation_rate': total_violations / max(len(self.violations), 1)
        }
    
    def generate_referee_report(self, time_window=300):
        """Generate a comprehensive referee report"""
        current_time = time.time()
        
        # Filter violations within time window
        recent_violations = [
            v for v in self.violations 
            if current_time - v.get('timestamp', 0) <= time_window
        ]
        
        if not recent_violations:
            return None
        
        # Analyze patterns
        offside_players = []
        foul_players = []
        
        for violation in recent_violations:
            if violation.get('offside'):
                offside_players.extend([p['player_id'] for p in violation['offside']])
            
            if violation.get('fouls'):
                foul_players.extend([f['player1'] for f in violation['fouls']])
                foul_players.extend([f['player2'] for f in violation['fouls']])
        
        # Count player violations
        from collections import Counter
        offside_counts = Counter(offside_players)
        foul_counts = Counter(foul_players)
        
        return {
            'time_window': time_window,
            'total_violations': len(recent_violations),
            'offside_analysis': {
                'total_offsides': len(offside_players),
                'players_with_offsides': dict(offside_counts),
                'most_offside_player': offside_counts.most_common(1)[0] if offside_counts else None
            },
            'foul_analysis': {
                'total_fouls': len(foul_players),
                'players_with_fouls': dict(foul_counts),
                'most_fouls_player': foul_counts.most_common(1)[0] if foul_counts else None
            },
            'recommendations': self.generate_referee_recommendations(recent_violations)
        }
    
    def generate_referee_recommendations(self, violations):
        """Generate recommendations for referees based on violations"""
        recommendations = []
        
        offside_count = sum(1 for v in violations if v.get('offside'))
        foul_count = sum(1 for v in violations if v.get('fouls'))
        
        if offside_count > 5:
            recommendations.append({
                'type': 'offside_attention',
                'message': 'High number of offside calls detected. Consider reviewing attacking line positioning.',
                'priority': 'high'
            })
        
        if foul_count > 10:
            recommendations.append({
                'type': 'foul_attention',
                'message': 'High number of fouls detected. Consider stricter enforcement of contact rules.',
                'priority': 'medium'
            })
        
        if not recommendations:
            recommendations.append({
                'type': 'normal_play',
                'message': 'Game proceeding normally with standard violation levels.',
                'priority': 'low'
            })
        
        return recommendations
    
    def calibrate_field_boundaries(self, field_corners):
        """Calibrate field boundaries for more accurate detection"""
        if len(field_corners) != 4:
            return False
        
        # Store field boundaries for more accurate detection
        self.field_boundaries = {
            'corners': field_corners,
            'width': max(corner[0] for corner in field_corners) - min(corner[0] for corner in field_corners),
            'height': max(corner[1] for corner in field_corners) - min(corner[1] for corner in field_corners)
        }
        
        return True
    
    def get_decision_confidence(self, decision_type, decision_data):
        """Calculate confidence level for a referee decision"""
        confidence = 0.5  # Base confidence
        
        if decision_type == 'offside':
            # Higher confidence if multiple players are offside
            if len(decision_data) > 1:
                confidence += 0.2
            
            # Higher confidence if ball is clearly in attacking half
            if decision_data and 'ball_position' in decision_data[0]:
                confidence += 0.1
        
        elif decision_type == 'foul':
            # Higher confidence for very close contact
            if decision_data and decision_data[0].get('distance', 0) < 20:
                confidence += 0.3
        
        elif decision_type == 'goal':
            # High confidence for goal events
            confidence = 0.9
        
        return min(confidence, 1.0)
    
    def reset(self):
        """Reset all referee assistant data"""
        self.violations.clear()
        self.offside_history.clear()
        self.foul_detections.clear()
        self.field_boundaries = None 
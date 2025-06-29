import numpy as np
import time
from collections import defaultdict
from scipy import stats
from scipy.spatial.distance import cdist
import cv2

class PerformanceAnalyzer:
    def __init__(self):
        self.player_metrics = defaultdict(dict)
        self.team_metrics = {}
        self.sprint_threshold = 150  # pixels per second
        self.analysis_window = 30  # seconds
        self.fatigue_data = defaultdict(list)
        self.tactical_efficiency = defaultdict(dict)
        self.biomechanical_data = defaultdict(list)
        
    def calculate_player_metrics(self, player_id, trajectory, duration):
        """Calculate comprehensive performance metrics for a player"""
        if not trajectory or len(trajectory) < 2:
            return None
        
        # Calculate distances and speeds
        distances = []
        speeds = []
        accelerations = []
        velocities = []
        
        for i in range(1, len(trajectory)):
            prev_pos = trajectory[i-1]
            curr_pos = trajectory[i]
            
            # Calculate distance
            distance = np.sqrt(
                (curr_pos[0] - prev_pos[0])**2 + 
                (curr_pos[1] - prev_pos[1])**2
            )
            distances.append(distance)
            
            # Calculate speed (assuming 30 FPS)
            speed = distance * 30  # pixels per second
            speeds.append(speed)
            
            # Calculate velocity vector
            velocity = [curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1]]
            velocities.append(velocity)
            
            # Calculate acceleration
            if i > 1:
                prev_speed = speeds[i-2]
                acceleration = (speed - prev_speed) * 30  # pixels per second squared
                accelerations.append(acceleration)
        
        if not speeds:
            return None
        
        # Calculate basic metrics
        total_distance = sum(distances)
        avg_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        min_speed = np.min(speeds)
        
        # Calculate sprint metrics
        sprints = [speed for speed in speeds if speed > self.sprint_threshold]
        sprint_count = len(sprints)
        sprint_distance = sum(sprints) / 30  # Convert to distance
        sprint_percentage = (sprint_count / len(speeds)) * 100 if speeds else 0
        
        # Calculate acceleration metrics
        avg_acceleration = np.mean(accelerations) if accelerations else 0
        max_acceleration = np.max(accelerations) if accelerations else 0
        min_acceleration = np.min(accelerations) if accelerations else 0
        
        # Calculate work rate (distance per minute)
        work_rate = (total_distance / duration) * 60 if duration > 0 else 0
        
        # Calculate intensity zones
        low_intensity = sum(1 for speed in speeds if speed < 50)
        medium_intensity = sum(1 for speed in speeds if 50 <= speed <= self.sprint_threshold)
        high_intensity = sum(1 for speed in speeds if speed > self.sprint_threshold)
        
        total_frames = len(speeds)
        intensity_distribution = {
            'low': (low_intensity / total_frames) * 100 if total_frames > 0 else 0,
            'medium': (medium_intensity / total_frames) * 100 if total_frames > 0 else 0,
            'high': (high_intensity / total_frames) * 100 if total_frames > 0 else 0
        }
        
        # Advanced metrics
        fatigue_index = self.calculate_fatigue_index(speeds, duration)
        tactical_efficiency = self.calculate_tactical_efficiency(trajectory, velocities)
        biomechanical_metrics = self.calculate_biomechanical_metrics(trajectory, speeds, accelerations)
        
        metrics = {
            'player_id': player_id,
            'total_distance': total_distance,
            'average_speed': avg_speed,
            'max_speed': max_speed,
            'min_speed': min_speed,
            'sprint_count': sprint_count,
            'sprint_distance': sprint_distance,
            'sprint_percentage': sprint_percentage,
            'average_acceleration': avg_acceleration,
            'max_acceleration': max_acceleration,
            'min_acceleration': min_acceleration,
            'work_rate': work_rate,
            'intensity_distribution': intensity_distribution,
            'fatigue_index': fatigue_index,
            'tactical_efficiency': tactical_efficiency,
            'biomechanical_metrics': biomechanical_metrics,
            'duration': duration,
            'timestamp': time.time()
        }
        
        # Store metrics
        self.player_metrics[player_id] = metrics
        
        return metrics
    
    def calculate_fatigue_index(self, speeds, duration):
        """Calculate fatigue index based on speed decline over time"""
        if len(speeds) < 10:
            return 0.0
        
        # Split speeds into quarters
        quarter_size = len(speeds) // 4
        quarters = [
            np.mean(speeds[:quarter_size]),
            np.mean(speeds[quarter_size:2*quarter_size]),
            np.mean(speeds[2*quarter_size:3*quarter_size]),
            np.mean(speeds[3*quarter_size:])
        ]
        
        # Calculate fatigue as decline in average speed
        initial_speed = quarters[0]
        final_speed = quarters[-1]
        
        if initial_speed > 0:
            fatigue_index = (initial_speed - final_speed) / initial_speed
            return max(0, min(1, fatigue_index))  # Normalize to 0-1
        else:
            return 0.0
    
    def calculate_tactical_efficiency(self, trajectory, velocities):
        """Calculate tactical efficiency based on movement patterns"""
        if len(trajectory) < 5:
            return 0.0
        
        # Calculate movement efficiency (straightness of path)
        total_distance = 0
        direct_distance = 0
        
        for i in range(1, len(trajectory)):
            # Actual distance traveled
            actual_dist = np.sqrt(
                (trajectory[i][0] - trajectory[i-1][0])**2 + 
                (trajectory[i][1] - trajectory[i-1][1])**2
            )
            total_distance += actual_dist
        
        # Direct distance from start to end
        direct_distance = np.sqrt(
            (trajectory[-1][0] - trajectory[0][0])**2 + 
            (trajectory[-1][1] - trajectory[0][1])**2
        )
        
        # Efficiency is direct distance / total distance
        if total_distance > 0:
            efficiency = direct_distance / total_distance
            return efficiency
        else:
            return 0.0
    
    def calculate_biomechanical_metrics(self, trajectory, speeds, accelerations):
        """Calculate advanced biomechanical metrics"""
        if len(trajectory) < 3:
            return {}
        
        # Calculate change of direction (COD) frequency
        cod_count = 0
        cod_threshold = 45  # degrees
        
        for i in range(1, len(trajectory) - 1):
            # Calculate angle between consecutive movement vectors
            vec1 = [trajectory[i][0] - trajectory[i-1][0], 
                   trajectory[i][1] - trajectory[i-1][1]]
            vec2 = [trajectory[i+1][0] - trajectory[i][0], 
                   trajectory[i+1][1] - trajectory[i][1]]
            
            # Calculate angle
            dot_product = vec1[0]*vec2[0] + vec1[1]*vec2[1]
            mag1 = np.sqrt(vec1[0]**2 + vec1[1]**2)
            mag2 = np.sqrt(vec2[0]**2 + vec2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                angle = np.arccos(cos_angle) * 180 / np.pi
                
                if angle > cod_threshold:
                    cod_count += 1
        
        # Calculate speed variability
        speed_variability = np.std(speeds) if speeds else 0
        
        # Calculate acceleration variability
        accel_variability = np.std(accelerations) if accelerations else 0
        
        # Calculate movement smoothness (inverse of jerk)
        jerk = 0
        if len(accelerations) > 1:
            for i in range(1, len(accelerations)):
                jerk += abs(accelerations[i] - accelerations[i-1])
            jerk = jerk / (len(accelerations) - 1)
        
        smoothness = 1 / (1 + jerk) if jerk > 0 else 1.0
        
        return {
            'cod_frequency': cod_count / max(len(trajectory), 1),
            'speed_variability': speed_variability,
            'acceleration_variability': accel_variability,
            'movement_smoothness': smoothness,
            'total_cod': cod_count
        }
    
    def calculate_team_metrics(self, all_player_metrics, ball_data=None):
        """Calculate team-level performance metrics"""
        if not all_player_metrics:
            return None
        
        # Aggregate player metrics
        total_distance = sum(metrics.get('total_distance', 0) for metrics in all_player_metrics.values())
        avg_speed = np.mean([metrics.get('average_speed', 0) for metrics in all_player_metrics.values()])
        total_sprints = sum(metrics.get('sprint_count', 0) for metrics in all_player_metrics.values())
        
        # Calculate team work rate
        total_duration = max(metrics.get('duration', 0) for metrics in all_player_metrics.values())
        team_work_rate = (total_distance / total_duration) * 60 if total_duration > 0 else 0
        
        # Calculate team intensity
        team_intensity = np.mean([
            metrics.get('intensity_distribution', {}).get('high', 0) 
            for metrics in all_player_metrics.values()
        ])
        
        # Ball possession metrics
        possession_metrics = self.calculate_possession_metrics(ball_data) if ball_data else {}
        
        team_metrics = {
            'total_distance': total_distance,
            'average_speed': avg_speed,
            'total_sprints': total_sprints,
            'team_work_rate': team_work_rate,
            'team_intensity': team_intensity,
            'player_count': len(all_player_metrics),
            'possession': possession_metrics,
            'timestamp': time.time()
        }
        
        self.team_metrics = team_metrics
        return team_metrics
    
    def calculate_possession_metrics(self, ball_data):
        """Calculate ball possession metrics"""
        if not ball_data:
            return {}
        
        # This would be implemented based on ball tracking data
        # For now, return placeholder metrics
        return {
            'possession_percentage': 50.0,  # Would be calculated from actual data
            'ball_touches': 0,
            'passes_completed': 0,
            'passes_attempted': 0,
            'pass_accuracy': 0.0
        }
    
    def get_performance_metrics(self, tracking_data):
        """Get comprehensive performance metrics from tracking data"""
        if not tracking_data or not tracking_data.get('frames'):
            return None
        
        # Extract player trajectories
        player_trajectories = {}
        frame_timestamps = tracking_data.get('timestamps', [])
        
        # Group player positions by player ID
        for frame_data in tracking_data['frames']:
            timestamp = frame_data.get('timestamp', 0)
            players = frame_data.get('players', [])
            
            for player in players:
                player_id = player['id']
                if player_id not in player_trajectories:
                    player_trajectories[player_id] = []
                
                player_trajectories[player_id].append({
                    'position': player['center'],
                    'timestamp': timestamp
                })
        
        # Calculate metrics for each player
        all_player_metrics = {}
        for player_id, trajectory_data in player_trajectories.items():
            if len(trajectory_data) < 2:
                continue
            
            # Extract positions and calculate duration
            positions = [entry['position'] for entry in trajectory_data]
            duration = trajectory_data[-1]['timestamp'] - trajectory_data[0]['timestamp']
            
            metrics = self.calculate_player_metrics(player_id, positions, duration)
            if metrics:
                all_player_metrics[player_id] = metrics
        
        # Calculate advanced team metrics
        team_metrics = self.calculate_advanced_team_metrics(all_player_metrics, tracking_data.get('ball'))
        if team_metrics is None:
            team_metrics = {
                'total_distance': 0,
                'average_speed': 0,
                'total_sprints': 0,
                'team_work_rate': 0,
                'team_intensity': 0,
                'team_fatigue': 0,
                'team_tactical_efficiency': 0,
                'player_count': 0,
                'possession': {},
                'timestamp': time.time()
            }
        
        # Calculate team coordination
        team_coordination = self.analyze_team_coordination(all_player_metrics, tracking_data)
        
        return {
            'individual_metrics': all_player_metrics,
            'team_metrics': team_metrics,
            'team_coordination': team_coordination,
            'timestamp': time.time()
        }
    
    def get_player_comparison(self, player_ids):
        """Compare performance metrics between players"""
        if not player_ids or len(player_ids) < 2:
            return None
        
        comparison = {}
        for player_id in player_ids:
            if player_id in self.player_metrics:
                comparison[player_id] = self.player_metrics[player_id]
        
        if len(comparison) < 2:
            return None
        
        # Calculate relative performance
        metrics_to_compare = ['total_distance', 'average_speed', 'sprint_count', 'work_rate']
        
        for metric in metrics_to_compare:
            values = [comparison[pid].get(metric, 0) for pid in comparison.keys()]
            if values:
                max_value = max(values)
                for player_id in comparison:
                    if max_value > 0:
                        comparison[player_id][f'{metric}_percentage'] = (
                            comparison[player_id].get(metric, 0) / max_value
                        ) * 100
        
        return comparison
    
    def get_performance_trends(self, player_id, time_window=300):
        """Get performance trends for a player over time"""
        if player_id not in self.player_metrics:
            return None
        
        # This would track performance over time windows
        # For now, return current metrics
        current_time = time.time()
        metrics = self.player_metrics[player_id]
        
        return {
            'player_id': player_id,
            'current_metrics': metrics,
            'trend': 'stable',  # Would be calculated from historical data
            'improvement_areas': self.identify_improvement_areas(metrics),
            'timestamp': current_time
        }
    
    def identify_improvement_areas(self, metrics):
        """Identify areas where player can improve"""
        improvement_areas = []
        
        # Define thresholds for different metrics
        thresholds = {
            'sprint_percentage': 15,  # Should be above 15%
            'work_rate': 1000,  # Should be above 1000 pixels/min
            'average_speed': 50  # Should be above 50 pixels/sec
        }
        
        for metric, threshold in thresholds.items():
            if metrics.get(metric, 0) < threshold:
                improvement_areas.append({
                    'metric': metric,
                    'current_value': metrics.get(metric, 0),
                    'target_value': threshold,
                    'improvement_needed': threshold - metrics.get(metric, 0)
                })
        
        return improvement_areas
    
    def generate_performance_report(self, player_id):
        """Generate a comprehensive performance report"""
        if player_id not in self.player_metrics:
            return None
        
        metrics = self.player_metrics[player_id]
        trends = self.get_performance_trends(player_id)
        improvement_areas = self.identify_improvement_areas(metrics)
        
        # Calculate performance score (0-100)
        performance_score = self.calculate_performance_score(metrics)
        
        return {
            'player_id': player_id,
            'performance_score': performance_score,
            'metrics': metrics,
            'trends': trends,
            'improvement_areas': improvement_areas,
            'recommendations': self.generate_recommendations(metrics, improvement_areas),
            'timestamp': time.time()
        }
    
    def calculate_performance_score(self, metrics):
        """Calculate overall performance score (0-100)"""
        score = 0
        
        # Distance contribution (25 points)
        distance_score = min(metrics.get('total_distance', 0) / 1000, 1) * 25
        score += distance_score
        
        # Speed contribution (25 points)
        speed_score = min(metrics.get('average_speed', 0) / 100, 1) * 25
        score += speed_score
        
        # Sprint contribution (25 points)
        sprint_score = min(metrics.get('sprint_percentage', 0) / 20, 1) * 25
        score += sprint_score
        
        # Work rate contribution (25 points)
        work_rate_score = min(metrics.get('work_rate', 0) / 1500, 1) * 25
        score += work_rate_score
        
        return min(score, 100)
    
    def generate_recommendations(self, metrics, improvement_areas):
        """Generate training recommendations based on metrics"""
        recommendations = []
        
        for area in improvement_areas:
            metric = area['metric']
            
            if metric == 'sprint_percentage':
                recommendations.append({
                    'area': 'Sprint Training',
                    'recommendation': 'Increase high-intensity training sessions',
                    'target': f"Increase sprint percentage to {area['target_value']}%"
                })
            elif metric == 'work_rate':
                recommendations.append({
                    'area': 'Endurance Training',
                    'recommendation': 'Focus on distance and stamina training',
                    'target': f"Increase work rate to {area['target_value']} pixels/min"
                })
            elif metric == 'average_speed':
                recommendations.append({
                    'area': 'Speed Training',
                    'recommendation': 'Include more speed drills and interval training',
                    'target': f"Increase average speed to {area['target_value']} pixels/sec"
                })
        
        return recommendations
    
    def reset(self):
        """Reset all performance data"""
        self.player_metrics.clear()
        self.team_metrics.clear()
        self.fatigue_data.clear()
        self.tactical_efficiency.clear()
        self.biomechanical_data.clear()
    
    def analyze_team_coordination(self, all_player_metrics, tracking_data):
        """Analyze team coordination and synchronization"""
        if not all_player_metrics or len(all_player_metrics) < 2:
            return {}
        
        # Calculate team synchronization
        player_speeds = []
        player_positions = []
        
        for player_id, metrics in all_player_metrics.items():
            if 'average_speed' in metrics:
                player_speeds.append(metrics['average_speed'])
            
            # Get player positions from tracking data
            if tracking_data and 'frames' in tracking_data:
                latest_frame = tracking_data['frames'][-1]
                for player in latest_frame.get('players', []):
                    if player['id'] == player_id:
                        player_positions.append(player['center'])
                        break
        
        # Calculate speed synchronization
        speed_sync = 1 - (np.std(player_speeds) / np.mean(player_speeds)) if player_speeds and np.mean(player_speeds) > 0 else 0
        
        # Calculate spatial coordination
        spatial_coordination = 0
        if len(player_positions) >= 2:
            # Calculate average distance between players
            distances = []
            for i in range(len(player_positions)):
                for j in range(i+1, len(player_positions)):
                    dist = np.sqrt(
                        (player_positions[i][0] - player_positions[j][0])**2 + 
                        (player_positions[i][1] - player_positions[j][1])**2
                    )
                    distances.append(dist)
            
            if distances:
                avg_distance = np.mean(distances)
                # Normalize coordination (closer players = better coordination)
                spatial_coordination = 1 / (1 + avg_distance / 1000)  # Normalize by 1000 pixels
        
        return {
            'speed_synchronization': max(0, min(1, speed_sync)),
            'spatial_coordination': spatial_coordination,
            'team_cohesion': (speed_sync + spatial_coordination) / 2
        }
    
    def calculate_advanced_team_metrics(self, all_player_metrics, ball_data=None):
        """Calculate advanced team-level performance metrics"""
        if not all_player_metrics:
            return None
        
        # Aggregate player metrics
        total_distance = sum(metrics.get('total_distance', 0) for metrics in all_player_metrics.values())
        avg_speed = np.mean([metrics.get('average_speed', 0) for metrics in all_player_metrics.values()])
        total_sprints = sum(metrics.get('sprint_count', 0) for metrics in all_player_metrics.values())
        
        # Calculate team work rate
        total_duration = max(metrics.get('duration', 0) for metrics in all_player_metrics.values())
        team_work_rate = (total_distance / total_duration) * 60 if total_duration > 0 else 0
        
        # Calculate team intensity
        team_intensity = np.mean([
            metrics.get('intensity_distribution', {}).get('high', 0) 
            for metrics in all_player_metrics.values()
        ])
        
        # Calculate team fatigue
        team_fatigue = np.mean([
            metrics.get('fatigue_index', 0) 
            for metrics in all_player_metrics.values()
        ])
        
        # Calculate team tactical efficiency
        team_tactical_efficiency = np.mean([
            metrics.get('tactical_efficiency', 0) 
            for metrics in all_player_metrics.values()
        ])
        
        # Ball possession metrics
        possession_metrics = self.calculate_possession_metrics(ball_data) if ball_data else {}
        
        team_metrics = {
            'total_distance': total_distance,
            'average_speed': avg_speed,
            'total_sprints': total_sprints,
            'team_work_rate': team_work_rate,
            'team_intensity': team_intensity,
            'team_fatigue': team_fatigue,
            'team_tactical_efficiency': team_tactical_efficiency,
            'player_count': len(all_player_metrics),
            'possession': possession_metrics,
            'timestamp': time.time()
        }
        
        self.team_metrics = team_metrics
        return team_metrics 
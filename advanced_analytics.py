import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import silhouette_score
import cv2
from scipy import stats
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

class AdvancedAnalytics:
    def __init__(self):
        self.player_clusters = {}
        self.team_patterns = {}
        self.anomaly_detector = None
        self.prediction_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        
    def analyze_player_clustering(self, tracking_data, n_clusters=3):
        """Use machine learning to cluster players based on movement patterns"""
        if not tracking_data or not tracking_data.get('frames'):
            return None
        
        # Extract player movement features
        player_features = self.extract_player_features(tracking_data)
        
        if len(player_features) < n_clusters:
            return None
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(player_features)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(features_scaled)
        
        # Calculate cluster quality
        silhouette_avg = silhouette_score(features_scaled, clusters)
        
        # Analyze cluster characteristics
        cluster_analysis = self.analyze_cluster_characteristics(player_features, clusters, kmeans.cluster_centers_)
        
        return {
            'clusters': clusters.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'silhouette_score': silhouette_avg,
            'cluster_analysis': cluster_analysis,
            'player_features': player_features.tolist()
        }
    
    def extract_player_features(self, tracking_data):
        """Extract numerical features from player tracking data"""
        features = []
        player_ids = []
        
        # Group data by player
        player_data = defaultdict(list)
        for frame in tracking_data['frames']:
            for player in frame.get('players', []):
                player_data[player['id']].append({
                    'position': player['center'],
                    'timestamp': frame.get('timestamp', 0)
                })
        
        for player_id, trajectory in player_data.items():
            if len(trajectory) < 5:
                continue
            
            # Calculate movement features
            positions = np.array([p['position'] for p in trajectory])
            
            # Distance features
            total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            avg_speed = total_distance / (len(trajectory) - 1) if len(trajectory) > 1 else 0
            
            # Position features
            x_positions = positions[:, 0]
            y_positions = positions[:, 1]
            
            # Spatial features
            x_range = np.max(x_positions) - np.min(x_positions)
            y_range = np.max(y_positions) - np.min(y_positions)
            spatial_coverage = x_range * y_range
            
            # Movement pattern features
            direction_changes = self.calculate_direction_changes(positions)
            movement_consistency = self.calculate_movement_consistency(positions)
            
            # Combine features
            player_feature = [
                total_distance,
                avg_speed,
                x_range,
                y_range,
                spatial_coverage,
                direction_changes,
                movement_consistency,
                np.mean(x_positions),
                np.mean(y_positions),
                np.std(x_positions),
                np.std(y_positions)
            ]
            
            features.append(player_feature)
            player_ids.append(player_id)
        
        return np.array(features) if features else np.array([])
    
    def calculate_direction_changes(self, positions):
        """Calculate number of direction changes in trajectory"""
        if len(positions) < 3:
            return 0
        
        direction_changes = 0
        for i in range(1, len(positions) - 1):
            vec1 = positions[i] - positions[i-1]
            vec2 = positions[i+1] - positions[i]
            
            # Calculate angle between vectors
            dot_product = np.dot(vec1, vec2)
            mag1 = np.linalg.norm(vec1)
            mag2 = np.linalg.norm(vec2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle) * 180 / np.pi
                
                if angle > 45:  # Significant direction change
                    direction_changes += 1
        
        return direction_changes
    
    def calculate_movement_consistency(self, positions):
        """Calculate consistency of movement patterns"""
        if len(positions) < 3:
            return 0
        
        # Calculate speed consistency
        speeds = []
        for i in range(1, len(positions)):
            speed = np.linalg.norm(positions[i] - positions[i-1])
            speeds.append(speed)
        
        # Consistency is inverse of speed variance
        speed_std = np.std(speeds)
        speed_mean = np.mean(speeds)
        
        if speed_mean > 0:
            consistency = 1 / (1 + speed_std / speed_mean)
            return consistency
        else:
            return 0
    
    def analyze_cluster_characteristics(self, features, clusters, centers):
        """Analyze characteristics of each cluster"""
        cluster_analysis = {}
        
        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            cluster_features = features[cluster_mask]
            cluster_center = centers[cluster_id]
            
            # Calculate cluster statistics
            cluster_stats = {
                'size': np.sum(cluster_mask),
                'avg_total_distance': np.mean(cluster_features[:, 0]),
                'avg_speed': np.mean(cluster_features[:, 1]),
                'avg_spatial_coverage': np.mean(cluster_features[:, 4]),
                'avg_direction_changes': np.mean(cluster_features[:, 5]),
                'avg_movement_consistency': np.mean(cluster_features[:, 6])
            }
            
            # Determine cluster type based on characteristics
            cluster_type = self.classify_cluster_type(cluster_stats)
            
            cluster_analysis[cluster_id] = {
                'stats': cluster_stats,
                'type': cluster_type,
                'center': cluster_center.tolist()
            }
        
        return cluster_analysis
    
    def classify_cluster_type(self, stats):
        """Classify cluster based on movement characteristics"""
        if stats['avg_speed'] > 100 and stats['avg_direction_changes'] > 5:
            return "high_activity_playmaker"
        elif stats['avg_spatial_coverage'] > 50000 and stats['avg_total_distance'] > 1000:
            return "wide_runner"
        elif stats['avg_movement_consistency'] > 0.7 and stats['avg_direction_changes'] < 3:
            return "positional_player"
        elif stats['avg_speed'] < 50 and stats['avg_spatial_coverage'] < 20000:
            return "static_player"
        else:
            return "balanced_player"
    
    def detect_anomalies(self, tracking_data, contamination=0.1):
        """Detect anomalous player movements using isolation forest"""
        if not tracking_data or not tracking_data.get('frames'):
            return None
        
        # Extract features for anomaly detection
        player_features = self.extract_player_features(tracking_data)
        
        if len(player_features) < 3:
            return None
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(player_features)
        
        # Train isolation forest
        self.anomaly_detector = IsolationForest(contamination=contamination, random_state=42)
        anomaly_scores = self.anomaly_detector.fit_predict(features_scaled)
        
        # Identify anomalies
        anomalies = anomaly_scores == -1
        
        return {
            'anomaly_scores': anomaly_scores.tolist(),
            'anomalies_detected': anomalies.tolist(),
            'anomaly_count': np.sum(anomalies),
            'total_players': len(anomalies)
        }
    
    def predict_performance_trends(self, historical_data, prediction_horizon=5):
        """Predict future performance trends using time series analysis"""
        if not historical_data or len(historical_data) < 10:
            return None
        
        # Prepare time series data
        time_series = self.prepare_time_series_data(historical_data)
        
        if time_series is None:
            return None
        
        # Simple moving average prediction
        predictions = self.simple_moving_average_prediction(time_series, prediction_horizon)
        
        # Trend analysis
        trend_analysis = self.analyze_trends(time_series)
        
        return {
            'predictions': predictions,
            'trend_analysis': trend_analysis,
            'prediction_horizon': prediction_horizon
        }
    
    def prepare_time_series_data(self, historical_data):
        """Prepare time series data for prediction"""
        try:
            # Extract performance metrics over time
            time_series_data = []
            
            for timestamp, data in historical_data.items():
                if 'team_metrics' in data:
                    metrics = data['team_metrics']
                    time_series_data.append({
                        'timestamp': timestamp,
                        'work_rate': metrics.get('team_work_rate', 0),
                        'intensity': metrics.get('team_intensity', 0),
                        'fatigue': metrics.get('team_fatigue', 0)
                    })
            
            if len(time_series_data) < 5:
                return None
            
            # Sort by timestamp
            time_series_data.sort(key=lambda x: x['timestamp'])
            
            return time_series_data
            
        except Exception as e:
            print(f"Error preparing time series data: {e}")
            return None
    
    def simple_moving_average_prediction(self, time_series, horizon):
        """Simple moving average prediction"""
        if len(time_series) < 3:
            return None
        
        # Calculate moving averages for different metrics
        work_rates = [d['work_rate'] for d in time_series]
        intensities = [d['intensity'] for d in time_series]
        fatigues = [d['fatigue'] for d in time_series]
        
        # Use last 3 values for prediction
        window_size = min(3, len(time_series))
        
        predictions = []
        for i in range(horizon):
            pred_work_rate = np.mean(work_rates[-window_size:])
            pred_intensity = np.mean(intensities[-window_size:])
            pred_fatigue = np.mean(fatigues[-window_size:])
            
            predictions.append({
                'step': i + 1,
                'predicted_work_rate': pred_work_rate,
                'predicted_intensity': pred_intensity,
                'predicted_fatigue': pred_fatigue
            })
        
        return predictions
    
    def analyze_trends(self, time_series):
        """Analyze trends in performance data"""
        if len(time_series) < 3:
            return None
        
        work_rates = [d['work_rate'] for d in time_series]
        intensities = [d['intensity'] for d in time_series]
        fatigues = [d['fatigue'] for d in time_series]
        
        # Calculate trend slopes
        x = np.arange(len(time_series))
        
        work_rate_slope = np.polyfit(x, work_rates, 1)[0]
        intensity_slope = np.polyfit(x, intensities, 1)[0]
        fatigue_slope = np.polyfit(x, fatigues, 1)[0]
        
        # Determine trend direction
        def get_trend_direction(slope):
            if slope > 0.1:
                return "increasing"
            elif slope < -0.1:
                return "decreasing"
            else:
                return "stable"
        
        return {
            'work_rate_trend': get_trend_direction(work_rate_slope),
            'intensity_trend': get_trend_direction(intensity_slope),
            'fatigue_trend': get_trend_direction(fatigue_slope),
            'work_rate_slope': work_rate_slope,
            'intensity_slope': intensity_slope,
            'fatigue_slope': fatigue_slope
        }
    
    def generate_heatmap_visualization(self, tracking_data, grid_size=20):
        """Generate advanced heatmap visualization"""
        if not tracking_data or not tracking_data.get('frames'):
            return None
        
        # Create heatmap grid
        frame_width = 1920  # Default width
        frame_height = 1080  # Default height
        
        grid_width = frame_width // grid_size
        grid_height = frame_height // grid_size
        
        heatmap = np.zeros((grid_height, grid_width))
        
        # Accumulate player positions
        for frame in tracking_data['frames']:
            for player in frame.get('players', []):
                x, y = player['center']
                grid_x = min(int(x // grid_size), grid_width - 1)
                grid_y = min(int(y // grid_size), grid_height - 1)
                heatmap[grid_y, grid_x] += 1
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Apply Gaussian smoothing
        heatmap_smooth = cv2.GaussianBlur(heatmap, (5, 5), 0)
        
        return {
            'heatmap': heatmap_smooth.tolist(),
            'grid_size': grid_size,
            'frame_width': frame_width,
            'frame_height': frame_height,
            'max_activity': float(np.max(heatmap_smooth)),
            'total_activity': float(np.sum(heatmap_smooth))
        }
    
    def analyze_team_synchronization(self, tracking_data):
        """Analyze team synchronization patterns"""
        if not tracking_data or not tracking_data.get('frames'):
            return None
        
        # Extract team movement patterns
        team_movements = []
        
        for frame in tracking_data['frames']:
            players = frame.get('players', [])
            if len(players) >= 2:
                # Calculate team center of mass
                positions = np.array([p['center'] for p in players])
                team_center = np.mean(positions, axis=0)
                
                # Calculate team spread
                distances_from_center = np.linalg.norm(positions - team_center, axis=1)
                team_spread = np.std(distances_from_center)
                
                team_movements.append({
                    'timestamp': frame.get('timestamp', 0),
                    'team_center': team_center.tolist(),
                    'team_spread': float(team_spread),
                    'player_count': len(players)
                })
        
        if len(team_movements) < 2:
            return None
        
        # Calculate synchronization metrics
        spreads = [m['team_spread'] for m in team_movements]
        avg_spread = np.mean(spreads)
        spread_consistency = 1 / (1 + np.std(spreads))
        
        # Analyze team compactness trend
        compactness_trend = self.analyze_compactness_trend(team_movements)
        
        return {
            'average_team_spread': avg_spread,
            'spread_consistency': spread_consistency,
            'team_compactness_trend': compactness_trend,
            'movement_data': team_movements
        }
    
    def analyze_compactness_trend(self, team_movements):
        """Analyze trend in team compactness"""
        if len(team_movements) < 5:
            return "insufficient_data"
        
        spreads = [m['team_spread'] for m in team_movements]
        x = np.arange(len(spreads))
        
        # Calculate trend
        slope = np.polyfit(x, spreads, 1)[0]
        
        if slope > 1:
            return "increasing_spread"
        elif slope < -1:
            return "decreasing_spread"
        else:
            return "stable_compactness"
    
    def reset(self):
        """Reset all advanced analytics data"""
        self.player_clusters.clear()
        self.team_patterns.clear()
        self.anomaly_detector = None
        self.prediction_model = None 
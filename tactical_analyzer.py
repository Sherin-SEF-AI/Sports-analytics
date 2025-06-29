import numpy as np
from collections import defaultdict
import time
import cv2
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde

class TacticalAnalyzer:
    def __init__(self):
        self.formation_history = []
        self.player_positions_history = []
        self.heatmap_data = defaultdict(int)
        self.passing_patterns = []
        self.possession_data = []
        self.pressing_data = []
        self.transition_data = []
        self.set_piece_data = []
        
    def analyze_formation(self, player_positions, frame_width, frame_height):
        """Analyze team formation based on player positions"""
        if not player_positions or len(player_positions) < 4:
            return None
        
        # Normalize positions to field coordinates
        normalized_positions = []
        for player in player_positions:
            x, y = player['center']
            norm_x = x / frame_width
            norm_y = y / frame_height
            normalized_positions.append((norm_x, norm_y, player['id']))
        
        # Sort players by Y position (field position from back to front)
        normalized_positions.sort(key=lambda p: p[1])
        
        # Analyze formation zones with more sophisticated logic
        defenders = []
        midfielders = []
        forwards = []
        
        # Use clustering to better identify player roles
        positions = np.array([(p[0], p[1]) for p in normalized_positions])
        
        # K-means clustering for role assignment
        from sklearn.cluster import KMeans
        if len(positions) >= 3:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(positions)
            
            # Sort clusters by Y position (back to front)
            cluster_centers = kmeans.cluster_centers_
            cluster_order = np.argsort(cluster_centers[:, 1])
            
            for i, (pos, cluster) in enumerate(zip(normalized_positions, clusters)):
                if cluster == cluster_order[0]:  # Back cluster
                    defenders.append((pos[0], pos[1], pos[2]))
                elif cluster == cluster_order[1]:  # Middle cluster
                    midfielders.append((pos[0], pos[1], pos[2]))
                else:  # Front cluster
                    forwards.append((pos[0], pos[1], pos[2]))
        else:
            # Fallback to simple zone-based assignment
            for x, y, player_id in normalized_positions:
                if y < 0.33:  # Defensive third
                    defenders.append((x, y, player_id))
                elif y < 0.67:  # Middle third
                    midfielders.append((x, y, player_id))
                else:  # Attacking third
                    forwards.append((x, y, player_id))
        
        # Determine formation
        formation = self.determine_formation(len(defenders), len(midfielders), len(forwards))
        
        # Calculate formation confidence
        confidence = self.calculate_formation_confidence(defenders, midfielders, forwards)
        
        # Analyze formation compactness
        compactness = self.calculate_formation_compactness(defenders, midfielders, forwards)
        
        formation_data = {
            'formation': formation,
            'defenders': len(defenders),
            'midfielders': len(midfielders),
            'forwards': len(forwards),
            'confidence': confidence,
            'compactness': compactness,
            'timestamp': time.time(),
            'player_positions': normalized_positions,
            'defender_positions': defenders,
            'midfielder_positions': midfielders,
            'forward_positions': forwards
        }
        
        # Store formation history
        self.formation_history.append(formation_data)
        
        return formation_data
    
    def determine_formation(self, defenders, midfielders, forwards):
        """Determine formation based on player distribution"""
        # Common formations
        formations = {
            (4, 4, 2): "4-4-2",
            (4, 3, 3): "4-3-3",
            (3, 5, 2): "3-5-2",
            (4, 2, 4): "4-2-4",
            (3, 4, 3): "3-4-3",
            (5, 3, 2): "5-3-2",
            (4, 1, 5): "4-1-5",
            (3, 6, 1): "3-6-1"
        }
        
        player_distribution = (defenders, midfielders, forwards)
        
        if player_distribution in formations:
            return formations[player_distribution]
        else:
            return f"{defenders}-{midfielders}-{forwards}"
    
    def calculate_formation_confidence(self, defenders, midfielders, forwards):
        """Calculate confidence in formation detection"""
        total_players = len(defenders) + len(midfielders) + len(forwards)
        
        if total_players < 4:
            return 0.0
        
        # Higher confidence for balanced formations
        balance_score = 1.0 - abs(len(defenders) - len(forwards)) / total_players
        
        # Higher confidence for common formations
        common_formations = [(4, 4, 2), (4, 3, 3), (3, 5, 2)]
        formation_score = 1.0 if (len(defenders), len(midfielders), len(forwards)) in common_formations else 0.7
        
        return (balance_score + formation_score) / 2
    
    def calculate_formation_compactness(self, defenders, midfielders, forwards):
        """Calculate how compact the formation is"""
        all_positions = defenders + midfielders + forwards
        if len(all_positions) < 2:
            return 0.0
        
        positions = np.array([(p[0], p[1]) for p in all_positions])
        
        # Calculate center of mass
        center = np.mean(positions, axis=0)
        
        # Calculate average distance from center
        distances = np.linalg.norm(positions - center, axis=1)
        avg_distance = np.mean(distances)
        
        # Normalize by field size (assuming 1x1 normalized field)
        compactness = 1.0 - min(avg_distance, 1.0)
        
        return compactness
    
    def analyze_player_spacing(self, player_positions, frame_width, frame_height):
        """Analyze spacing between players"""
        if len(player_positions) < 2:
            return None
        
        distances = []
        spacing_analysis = {
            'average_distance': 0,
            'min_distance': 0,
            'max_distance': 0,
            'spacing_efficiency': 0,
            'closest_pairs': []
        }
        
        # Calculate all pairwise distances
        for i, player1 in enumerate(player_positions):
            for j, player2 in enumerate(player_positions[i+1:], i+1):
                pos1 = player1['center']
                pos2 = player2['center']
                
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                distances.append({
                    'distance': distance,
                    'player1': player1['id'],
                    'player2': player2['id'],
                    'positions': (pos1, pos2)
                })
        
        if distances:
            distances.sort(key=lambda x: x['distance'])
            
            spacing_analysis['average_distance'] = np.mean([d['distance'] for d in distances])
            spacing_analysis['min_distance'] = distances[0]['distance']
            spacing_analysis['max_distance'] = distances[-1]['distance']
            spacing_analysis['closest_pairs'] = distances[:3]  # Top 3 closest pairs
            
            # Calculate spacing efficiency (how well players are distributed)
            field_area = frame_width * frame_height
            optimal_distance = np.sqrt(field_area / len(player_positions))
            spacing_analysis['spacing_efficiency'] = 1.0 - abs(
                spacing_analysis['average_distance'] - optimal_distance) / optimal_distance
        
        return spacing_analysis
    
    def generate_heatmap(self, player_positions, frame_width, frame_height):
        """Generate heatmap data for player activity"""
        if not player_positions:
            return None
        
        # Create a grid for heatmap
        grid_size = 20
        grid_width = frame_width // grid_size
        grid_height = frame_height // grid_size
        
        heatmap = np.zeros((grid_height, grid_width))
        
        for player in player_positions:
            x, y = player['center']
            grid_x = min(int(x // grid_size), grid_width - 1)
            grid_y = min(int(y // grid_size), grid_height - 1)
            heatmap[grid_y, grid_x] += 1
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return {
            'heatmap': heatmap.tolist(),
            'grid_size': grid_size,
            'frame_width': frame_width,
            'frame_height': frame_height
        }
    
    def analyze_passing_patterns(self, ball_trajectory, player_positions):
        """Analyze passing patterns based on ball movement"""
        if not ball_trajectory or len(ball_trajectory) < 10:
            return None
        
        # Find potential passes (high-speed ball movements)
        passes = []
        for i in range(1, len(ball_trajectory)):
            prev_pos = ball_trajectory[i-1]['position']
            curr_pos = ball_trajectory[i]['position']
            
            # Calculate ball speed
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            speed = np.sqrt(dx*dx + dy*dy)
            
            # High speed indicates a pass
            if speed > 100:  # Threshold for pass detection
                # Find potential passer and receiver
                passer = self.find_closest_player(prev_pos, player_positions)
                receiver = self.find_closest_player(curr_pos, player_positions)
                
                if passer and receiver and passer['id'] != receiver['id']:
                    passes.append({
                        'passer': passer['id'],
                        'receiver': receiver['id'],
                        'speed': speed,
                        'start_pos': prev_pos,
                        'end_pos': curr_pos,
                        'timestamp': ball_trajectory[i]['time']
                    })
        
        return {
            'total_passes': len(passes),
            'passes': passes,
            'passing_network': self.build_passing_network(passes)
        }
    
    def find_closest_player(self, position, player_positions):
        """Find the closest player to a given position"""
        if not player_positions:
            return None
        
        closest_player = None
        min_distance = float('inf')
        
        for player in player_positions:
            player_pos = player['center']
            distance = np.sqrt(
                (position[0] - player_pos[0])**2 + 
                (position[1] - player_pos[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_player = player
        
        return closest_player if min_distance < 100 else None  # 100 pixel threshold
    
    def build_passing_network(self, passes):
        """Build passing network between players"""
        network = defaultdict(lambda: {'passes_to': defaultdict(int), 'total_passes': 0})
        
        for pass_data in passes:
            passer = pass_data['passer']
            receiver = pass_data['receiver']
            
            network[passer]['passes_to'][receiver] += 1
            network[passer]['total_passes'] += 1
        
        return dict(network)
    
    def analyze_possession(self, ball_position, player_positions, frame_width, frame_height):
        """Analyze ball possession and control"""
        if not ball_position or not player_positions:
            return None
        
        # Find closest player to ball
        ball_center = ball_position['center']
        closest_player = None
        min_distance = float('inf')
        
        for player in player_positions:
            player_center = player['center']
            distance = np.sqrt(
                (ball_center[0] - player_center[0])**2 + 
                (ball_center[1] - player_center[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_player = player
        
        # Determine possession based on distance threshold
        possession_threshold = 50  # pixels
        has_possession = min_distance < possession_threshold
        
        # Calculate possession quality (how close the ball is)
        possession_quality = max(0, 1 - (min_distance / possession_threshold))
        
        # Analyze field zones
        ball_x, ball_y = ball_center
        field_zone = self.get_field_zone(ball_x, ball_y, frame_width, frame_height)
        
        possession_data = {
            'has_possession': has_possession,
            'possessing_player': closest_player['id'] if closest_player else None,
            'possession_distance': min_distance,
            'possession_quality': possession_quality,
            'field_zone': field_zone,
            'timestamp': time.time()
        }
        
        self.possession_data.append(possession_data)
        return possession_data
    
    def get_field_zone(self, x, y, frame_width, frame_height):
        """Get the field zone where the ball/player is located"""
        norm_x = x / frame_width
        norm_y = y / frame_height
        
        # Define zones
        if norm_y < 0.33:
            vertical_zone = "defensive"
        elif norm_y < 0.67:
            vertical_zone = "midfield"
        else:
            vertical_zone = "attacking"
        
        if norm_x < 0.33:
            horizontal_zone = "left"
        elif norm_x < 0.67:
            horizontal_zone = "center"
        else:
            horizontal_zone = "right"
        
        return f"{vertical_zone}_{horizontal_zone}"
    
    def analyze_pressing_intensity(self, player_positions, ball_position, frame_width, frame_height):
        """Analyze pressing intensity and defensive pressure"""
        if not player_positions or not ball_position:
            return None
        
        ball_center = ball_position['center']
        pressing_radius = 150  # pixels
        
        # Count players in pressing radius
        pressing_players = []
        for player in player_positions:
            player_center = player['center']
            distance = np.sqrt(
                (ball_center[0] - player_center[0])**2 + 
                (ball_center[1] - player_center[1])**2
            )
            
            if distance <= pressing_radius:
                pressing_players.append({
                    'player_id': player['id'],
                    'distance': distance,
                    'pressure_intensity': 1 - (distance / pressing_radius)
                })
        
        # Calculate pressing metrics
        pressing_intensity = len(pressing_players) / max(len(player_positions), 1)
        avg_pressure = np.mean([p['pressure_intensity'] for p in pressing_players]) if pressing_players else 0
        
        # Determine pressing type
        if len(pressing_players) >= 3:
            pressing_type = "high_press"
        elif len(pressing_players) >= 2:
            pressing_type = "medium_press"
        else:
            pressing_type = "low_press"
        
        pressing_data = {
            'pressing_players': pressing_players,
            'pressing_intensity': pressing_intensity,
            'average_pressure': avg_pressure,
            'pressing_type': pressing_type,
            'players_in_radius': len(pressing_players),
            'timestamp': time.time()
        }
        
        self.pressing_data.append(pressing_data)
        return pressing_data
    
    def analyze_transitions(self, current_frame, previous_frame):
        """Analyze transition moments (attack to defense, defense to attack)"""
        if not previous_frame:
            return None
        
        current_ball = current_frame.get('ball')
        previous_ball = previous_frame.get('ball')
        
        if not current_ball or not previous_ball:
            return None
        
        # Calculate ball movement
        current_pos = current_ball['center']
        previous_pos = previous_ball['center']
        
        ball_movement = np.sqrt(
            (current_pos[0] - previous_pos[0])**2 + 
            (current_pos[1] - previous_pos[1])**2
        )
        
        # Detect transition based on ball movement and field position
        transition_threshold = 100  # pixels
        is_transition = ball_movement > transition_threshold
        
        if is_transition:
            # Determine transition type
            current_zone = self.get_field_zone(current_pos[0], current_pos[1], 1920, 1080)
            previous_zone = self.get_field_zone(previous_pos[0], previous_pos[1], 1920, 1080)
            
            if "defensive" in previous_zone and "attacking" in current_zone:
                transition_type = "counter_attack"
            elif "attacking" in previous_zone and "defensive" in current_zone:
                transition_type = "defensive_transition"
            else:
                transition_type = "lateral_transition"
            
            transition_data = {
                'is_transition': True,
                'transition_type': transition_type,
                'ball_movement': ball_movement,
                'from_zone': previous_zone,
                'to_zone': current_zone,
                'timestamp': time.time()
            }
            
            self.transition_data.append(transition_data)
            return transition_data
        
        return None
    
    def analyze_set_pieces(self, ball_position, player_positions, frame_width, frame_height):
        """Analyze set-piece situations (corners, free kicks, etc.)"""
        if not ball_position or not player_positions:
            return None
        
        ball_center = ball_position['center']
        norm_x = ball_center[0] / frame_width
        norm_y = ball_center[1] / frame_height
        
        # Detect corner situations
        corner_threshold = 0.1
        is_corner = (norm_x < corner_threshold or norm_x > (1 - corner_threshold)) and \
                   (norm_y < corner_threshold or norm_y > (1 - corner_threshold))
        
        # Detect free kick situations (ball near edge)
        edge_threshold = 0.15
        is_free_kick = (norm_x < edge_threshold or norm_x > (1 - edge_threshold) or 
                       norm_y < edge_threshold or norm_y > (1 - edge_threshold))
        
        if is_corner or is_free_kick:
            # Analyze player positioning for set piece
            attacking_players = []
            defending_players = []
            
            for player in player_positions:
                player_center = player['center']
                distance_to_ball = np.sqrt(
                    (ball_center[0] - player_center[0])**2 + 
                    (ball_center[1] - player_center[1])**2
                )
                
                # Players close to ball are likely attacking
                if distance_to_ball < 100:
                    attacking_players.append(player)
                else:
                    defending_players.append(player)
            
            set_piece_data = {
                'is_set_piece': True,
                'set_piece_type': 'corner' if is_corner else 'free_kick',
                'attacking_players': len(attacking_players),
                'defending_players': len(defending_players),
                'ball_position': ball_center,
                'timestamp': time.time()
            }
            
            self.set_piece_data.append(set_piece_data)
            return set_piece_data
        
        return None
    
    def get_tactical_insights(self, tracking_data):
        """Get comprehensive tactical insights"""
        if not tracking_data or not tracking_data.get('frames'):
            return None
        
        latest_frame = tracking_data['frames'][-1]
        player_positions = latest_frame.get('players', [])
        
        if not player_positions:
            return None
        
        # Analyze current formation
        formation_analysis = self.analyze_formation(
            player_positions, 
            1920, 1080  # Default frame size, should be passed from video processor
        )
        
        # Analyze player spacing
        spacing_analysis = self.analyze_player_spacing(
            player_positions,
            1920, 1080
        )
        
        # Generate heatmap
        heatmap_data = self.generate_heatmap(
            player_positions,
            1920, 1080
        )
        
        # Analyze passing patterns
        ball_trajectory = tracking_data.get('ball', [])
        passing_analysis = self.analyze_passing_patterns(ball_trajectory, player_positions)
        
        # Analyze possession
        ball_position = latest_frame.get('ball', {})
        possession_analysis = self.analyze_possession(ball_position, player_positions, 1920, 1080)
        
        # Analyze pressing intensity
        pressing_analysis = self.analyze_pressing_intensity(player_positions, ball_position, 1920, 1080)
        
        # Analyze transitions
        if len(tracking_data['frames']) > 1:
            transition_analysis = self.analyze_transitions(latest_frame, tracking_data['frames'][-2])
        else:
            transition_analysis = None
        
        # Analyze set pieces
        set_piece_analysis = self.analyze_set_pieces(ball_position, player_positions, 1920, 1080)
        
        return {
            'formation': formation_analysis,
            'spacing': spacing_analysis,
            'heatmap': heatmap_data,
            'passing': passing_analysis,
            'possession': possession_analysis,
            'pressing': pressing_analysis,
            'transition': transition_analysis,
            'set_piece': set_piece_analysis,
            'timestamp': time.time()
        }
    
    def get_formation_trends(self):
        """Analyze formation changes over time"""
        if len(self.formation_history) < 2:
            return None
        
        formations = [f['formation'] for f in self.formation_history]
        formation_counts = defaultdict(int)
        
        for formation in formations:
            formation_counts[formation] += 1
        
        most_common_formation = max(formation_counts.items(), key=lambda x: x[1])
        
        return {
            'most_common_formation': most_common_formation[0],
            'formation_frequency': dict(formation_counts),
            'formation_changes': len(set(formations)),
            'total_analyses': len(formations)
        }
    
    def reset(self):
        """Reset tactical analysis data"""
        self.formation_history.clear()
        self.player_positions_history.clear()
        self.heatmap_data.clear()
        self.passing_patterns.clear()
        self.possession_data.clear()
        self.pressing_data.clear()
        self.transition_data.clear()
        self.set_piece_data.clear() 
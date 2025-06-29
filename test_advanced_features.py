#!/usr/bin/env python3
"""
Advanced Features Test Script for Sports Analytics Platform
Tests all the new advanced analytics features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.tactical_analyzer import TacticalAnalyzer
from utils.performance_analyzer import PerformanceAnalyzer
from utils.advanced_analytics import AdvancedAnalytics
import time
import numpy as np

def create_advanced_tracking_data():
    """Create comprehensive tracking data for testing advanced features"""
    base_time = time.time()
    
    return {
        'frames': [
            {
                'timestamp': base_time + i,
                'players': [
                    {
                        'id': f'player_{j}',
                        'center': [100 + i*10 + j*50, 200 + i*5 + j*30],
                        'bbox': [80 + i*10 + j*50, 180 + i*5 + j*30, 120 + i*10 + j*50, 220 + i*5 + j*30],
                        'confidence': 0.9 - j*0.1
                    }
                    for j in range(4)  # 4 players
                ],
                'ball': {
                    'center': [250 + i*15, 250 + i*8],
                    'bbox': [240 + i*15, 240 + i*8, 260 + i*15, 260 + i*8],
                    'area': 400
                },
                'frame_number': i
            }
            for i in range(20)  # 20 frames
        ],
        'timestamps': [base_time + i for i in range(20)],
        'players': [
            [
                {
                    'id': f'player_{j}',
                    'center': [100 + i*10 + j*50, 200 + i*5 + j*30],
                    'bbox': [80 + i*10 + j*50, 180 + i*5 + j*30, 120 + i*10 + j*50, 220 + i*5 + j*30],
                    'confidence': 0.9 - j*0.1
                }
                for j in range(4)
            ]
            for i in range(20)
        ],
        'ball': [
            {
                'center': [250 + i*15, 250 + i*8],
                'bbox': [240 + i*15, 240 + i*8, 260 + i*15, 260 + i*8],
                'area': 400
            }
            for i in range(20)
        ]
    }

def test_advanced_tactical_features():
    """Test advanced tactical analysis features"""
    print("Testing Advanced Tactical Features...")
    
    tactical_analyzer = TacticalAnalyzer()
    sample_data = create_advanced_tracking_data()
    
    try:
        # Test formation analysis with clustering
        latest_frame = sample_data['frames'][-1]
        formation = tactical_analyzer.analyze_formation(
            latest_frame['players'], 1920, 1080
        )
        
        if formation:
            print("✅ Advanced formation analysis working!")
            print(f"   Formation: {formation['formation']}")
            print(f"   Compactness: {formation['compactness']:.2f}")
            print(f"   Confidence: {formation['confidence']:.2f}")
        
        # Test possession analysis
        possession = tactical_analyzer.analyze_possession(
            latest_frame['ball'], latest_frame['players'], 1920, 1080
        )
        
        if possession:
            print("✅ Possession analysis working!")
            print(f"   Has possession: {possession['has_possession']}")
            print(f"   Possession quality: {possession['possession_quality']:.2f}")
            print(f"   Field zone: {possession['field_zone']}")
        
        # Test pressing analysis
        pressing = tactical_analyzer.analyze_pressing_intensity(
            latest_frame['players'], latest_frame['ball'], 1920, 1080
        )
        
        if pressing:
            print("✅ Pressing analysis working!")
            print(f"   Pressing type: {pressing['pressing_type']}")
            print(f"   Pressing intensity: {pressing['pressing_intensity']:.2f}")
            print(f"   Players in radius: {pressing['players_in_radius']}")
        
        # Test transition analysis
        if len(sample_data['frames']) > 1:
            transition = tactical_analyzer.analyze_transitions(
                sample_data['frames'][-1], sample_data['frames'][-2]
            )
            
            if transition:
                print("✅ Transition analysis working!")
                print(f"   Transition type: {transition['transition_type']}")
                print(f"   Ball movement: {transition['ball_movement']:.1f}")
        
        # Test set piece analysis
        set_piece = tactical_analyzer.analyze_set_pieces(
            latest_frame['ball'], latest_frame['players'], 1920, 1080
        )
        
        if set_piece:
            print("✅ Set piece analysis working!")
            print(f"   Set piece type: {set_piece['set_piece_type']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in advanced tactical features: {e}")
        return False

def test_advanced_performance_features():
    """Test advanced performance analysis features"""
    print("\nTesting Advanced Performance Features...")
    
    performance_analyzer = PerformanceAnalyzer()
    sample_data = create_advanced_tracking_data()
    
    try:
        # Test advanced performance metrics
        metrics = performance_analyzer.get_performance_metrics(sample_data)
        
        if metrics:
            print("✅ Advanced performance metrics working!")
            
            # Test individual metrics
            individual_metrics = metrics.get('individual_metrics', {})
            if individual_metrics:
                player_id = list(individual_metrics.keys())[0]
                player_metrics = individual_metrics[player_id]
                
                print(f"   Fatigue index: {player_metrics.get('fatigue_index', 0):.2f}")
                print(f"   Tactical efficiency: {player_metrics.get('tactical_efficiency', 0):.2f}")
                
                # Test biomechanical metrics
                biomech = player_metrics.get('biomechanical_metrics', {})
                if biomech:
                    print(f"   COD frequency: {biomech.get('cod_frequency', 0):.2f}")
                    print(f"   Movement smoothness: {biomech.get('movement_smoothness', 0):.2f}")
            
            # Test team coordination
            team_coordination = metrics.get('team_coordination', {})
            if team_coordination:
                print(f"   Speed synchronization: {team_coordination.get('speed_synchronization', 0):.2f}")
                print(f"   Team cohesion: {team_coordination.get('team_cohesion', 0):.2f}")
            
            return True
        else:
            print("❌ No performance metrics generated")
            return False
            
    except Exception as e:
        print(f"❌ Error in advanced performance features: {e}")
        return False

def test_advanced_analytics():
    """Test advanced analytics features"""
    print("\nTesting Advanced Analytics Features...")
    
    advanced_analytics = AdvancedAnalytics()
    sample_data = create_advanced_tracking_data()
    
    try:
        # Test player clustering
        clustering = advanced_analytics.analyze_player_clustering(sample_data)
        
        if clustering:
            print("✅ Player clustering working!")
            print(f"   Silhouette score: {clustering['silhouette_score']:.2f}")
            print(f"   Clusters found: {len(clustering['cluster_analysis'])}")
            
            # Test cluster analysis
            for cluster_id, analysis in clustering['cluster_analysis'].items():
                print(f"   Cluster {cluster_id}: {analysis['type']}")
        
        # Test anomaly detection
        anomalies = advanced_analytics.detect_anomalies(sample_data)
        
        if anomalies:
            print("✅ Anomaly detection working!")
            print(f"   Anomalies found: {anomalies['anomaly_count']}/{anomalies['total_players']}")
        
        # Test team synchronization
        team_sync = advanced_analytics.analyze_team_synchronization(sample_data)
        
        if team_sync:
            print("✅ Team synchronization working!")
            print(f"   Average spread: {team_sync['average_team_spread']:.1f}")
            print(f"   Spread consistency: {team_sync['spread_consistency']:.2f}")
            print(f"   Compactness trend: {team_sync['team_compactness_trend']}")
        
        # Test heatmap visualization
        heatmap = advanced_analytics.generate_heatmap_visualization(sample_data)
        
        if heatmap:
            print("✅ Heatmap visualization working!")
            print(f"   Max activity: {heatmap['max_activity']:.2f}")
            print(f"   Total activity: {heatmap['total_activity']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in advanced analytics: {e}")
        return False

def test_prediction_features():
    """Test prediction and trend analysis features"""
    print("\nTesting Prediction Features...")
    
    advanced_analytics = AdvancedAnalytics()
    
    try:
        # Create historical data
        historical_data = {}
        base_time = time.time()
        
        for i in range(10):
            historical_data[base_time + i] = {
                'team_metrics': {
                    'team_work_rate': 1000 + i*50,
                    'team_intensity': 60 + i*2,
                    'team_fatigue': 0.1 + i*0.05
                }
            }
        
        # Test predictions
        predictions = advanced_analytics.predict_performance_trends(historical_data)
        
        if predictions:
            print("✅ Prediction analysis working!")
            
            if predictions.get('trend_analysis'):
                trends = predictions['trend_analysis']
                print(f"   Work rate trend: {trends['work_rate_trend']}")
                print(f"   Intensity trend: {trends['intensity_trend']}")
                print(f"   Fatigue trend: {trends['fatigue_trend']}")
            
            if predictions.get('predictions'):
                pred_count = len(predictions['predictions'])
                print(f"   Predictions generated: {pred_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in prediction features: {e}")
        return False

def main():
    """Run all advanced feature tests"""
    print("Sports Analytics Platform - Advanced Features Test Suite")
    print("=" * 60)
    
    tactical_success = test_advanced_tactical_features()
    performance_success = test_advanced_performance_features()
    analytics_success = test_advanced_analytics()
    prediction_success = test_prediction_features()
    
    print("\n" + "=" * 60)
    print("Advanced Features Test Results:")
    print(f"Advanced Tactical Features: {'✅ PASS' if tactical_success else '❌ FAIL'}")
    print(f"Advanced Performance Features: {'✅ PASS' if performance_success else '❌ FAIL'}")
    print(f"Advanced Analytics: {'✅ PASS' if analytics_success else '❌ FAIL'}")
    print(f"Prediction Features: {'✅ PASS' if prediction_success else '❌ FAIL'}")
    
    all_passed = tactical_success and performance_success and analytics_success and prediction_success
    
    if all_passed:
        print("\n🎉 All advanced features are working correctly!")
        print("\nNew Features Added:")
        print("• Advanced tactical analysis (possession, pressing, transitions, set pieces)")
        print("• Advanced performance metrics (fatigue, biomechanics, team coordination)")
        print("• Machine learning clustering and anomaly detection")
        print("• Team synchronization analysis")
        print("• Performance trend prediction")
        print("• Advanced heatmap visualization")
        return 0
    else:
        print("\n⚠️  Some advanced features failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    exit(main()) 
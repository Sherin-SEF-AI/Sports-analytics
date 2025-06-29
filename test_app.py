#!/usr/bin/env python3
"""
Test script for Sports Analytics Platform
Tests the tactical and performance analyzers with sample data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.tactical_analyzer import TacticalAnalyzer
from utils.performance_analyzer import PerformanceAnalyzer
import time

def create_sample_tracking_data():
    """Create sample tracking data for testing"""
    return {
        'frames': [
            {
                'timestamp': time.time(),
                'players': [
                    {
                        'id': 'player_1',
                        'center': [100, 200],
                        'bbox': [80, 180, 120, 220],
                        'confidence': 0.9
                    },
                    {
                        'id': 'player_2', 
                        'center': [300, 150],
                        'bbox': [280, 130, 320, 170],
                        'confidence': 0.8
                    },
                    {
                        'id': 'player_3',
                        'center': [500, 300],
                        'bbox': [480, 280, 520, 320],
                        'confidence': 0.85
                    },
                    {
                        'id': 'player_4',
                        'center': [200, 400],
                        'bbox': [180, 380, 220, 420],
                        'confidence': 0.75
                    }
                ],
                'ball': {
                    'center': [250, 250],
                    'bbox': [240, 240, 260, 260],
                    'area': 400
                },
                'frame_number': 1
            }
        ],
        'timestamps': [time.time()],
        'players': [
            [
                {
                    'id': 'player_1',
                    'center': [100, 200],
                    'bbox': [80, 180, 120, 220],
                    'confidence': 0.9
                },
                {
                    'id': 'player_2', 
                    'center': [300, 150],
                    'bbox': [280, 130, 320, 170],
                    'confidence': 0.8
                },
                {
                    'id': 'player_3',
                    'center': [500, 300],
                    'bbox': [480, 280, 520, 320],
                    'confidence': 0.85
                },
                {
                    'id': 'player_4',
                    'center': [200, 400],
                    'bbox': [180, 380, 220, 420],
                    'confidence': 0.75
                }
            ]
        ],
        'ball': [
            {
                'center': [250, 250],
                'bbox': [240, 240, 260, 260],
                'area': 400
            }
        ]
    }

def test_tactical_analyzer():
    """Test the tactical analyzer"""
    print("Testing Tactical Analyzer...")
    
    tactical_analyzer = TacticalAnalyzer()
    sample_data = create_sample_tracking_data()
    
    try:
        insights = tactical_analyzer.get_tactical_insights(sample_data)
        if insights:
            print("✅ Tactical insights generated successfully!")
            print(f"Formation: {insights.get('formation', {}).get('formation', 'Unknown')}")
            print(f"Spacing efficiency: {insights.get('spacing', {}).get('spacing_efficiency', 0):.2f}")
            return True
        else:
            print("❌ No tactical insights generated")
            return False
    except Exception as e:
        print(f"❌ Error in tactical analyzer: {e}")
        return False

def test_performance_analyzer():
    """Test the performance analyzer"""
    print("\nTesting Performance Analyzer...")
    
    performance_analyzer = PerformanceAnalyzer()
    sample_data = create_sample_tracking_data()
    
    try:
        metrics = performance_analyzer.get_performance_metrics(sample_data)
        if metrics:
            print("✅ Performance metrics generated successfully!")
            player_count = len(metrics.get('individual_metrics', {}))
            print(f"Players analyzed: {player_count}")
            team_work_rate = metrics.get('team_metrics', {}).get('team_work_rate', 0)
            print(f"Team work rate: {team_work_rate:.2f}")
            return True
        else:
            print("❌ No performance metrics generated")
            return False
    except Exception as e:
        print(f"❌ Error in performance analyzer: {e}")
        return False

def main():
    """Run all tests"""
    print("Sports Analytics Platform - Test Suite")
    print("=" * 50)
    
    tactical_success = test_tactical_analyzer()
    performance_success = test_performance_analyzer()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Tactical Analyzer: {'✅ PASS' if tactical_success else '❌ FAIL'}")
    print(f"Performance Analyzer: {'✅ PASS' if performance_success else '❌ FAIL'}")
    
    if tactical_success and performance_success:
        print("\n🎉 All tests passed! The analyzers are working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    exit(main()) 
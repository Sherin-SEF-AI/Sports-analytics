from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import json
import threading
import time
from datetime import datetime

# Import our custom modules
from utils.video_processor import VideoProcessor
from utils.player_tracker import PlayerTracker
from utils.ball_tracker import BallTracker
from utils.tactical_analyzer import TacticalAnalyzer
from utils.performance_analyzer import PerformanceAnalyzer
from utils.referee_assistant import RefereeAssistant
from utils.advanced_analytics import AdvancedAnalytics

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sports-analytics-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize SocketIO with CORS
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
CORS(app)

# Global variables for tracking
active_sessions = {}
video_processors = {}

# Initialize analytics modules
player_tracker = PlayerTracker()
ball_tracker = BallTracker()
tactical_analyzer = TacticalAnalyzer()
performance_analyzer = PerformanceAnalyzer()
referee_assistant = RefereeAssistant()
advanced_analytics = AdvancedAnalytics()

@app.route('/')
def index():
    """Main application page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(active_sessions)
    })

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video file uploads"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Generate session ID
        session_id = f"upload_{timestamp}"
        
        # Initialize video processor for uploaded file
        video_processor = VideoProcessor(
            video_path=filepath,
            session_id=session_id,
            socketio=socketio
        )
        video_processors[session_id] = video_processor
        
        return jsonify({
            'session_id': session_id,
            'filename': filename,
            'message': 'Video uploaded successfully'
        })

@app.route('/api/analysis/<session_id>')
def get_analysis(session_id):
    """Get analysis results for a session"""
    if session_id not in video_processors:
        return jsonify({'error': 'Session not found'}), 404
    
    processor = video_processors[session_id]
    analysis = processor.get_analysis_results()
    
    return jsonify(analysis)

@app.route('/api/sessions')
def list_sessions():
    """List all active sessions"""
    sessions = []
    for session_id, processor in video_processors.items():
        sessions.append({
            'session_id': session_id,
            'status': processor.get_status(),
            'created_at': processor.created_at.isoformat()
        })
    
    return jsonify(sessions)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('connected', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")
    # Clean up any active sessions for this client
    for session_id in list(active_sessions.keys()):
        if active_sessions[session_id].get('client_id') == request.sid:
            cleanup_session(session_id)

@socketio.on('start_live_analysis')
def handle_start_live_analysis(data):
    """Start live video analysis"""
    try:
        camera_index = data.get('camera_index', 0)
        session_id = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize live video processor
        video_processor = VideoProcessor(
            camera_index=camera_index,
            session_id=session_id,
            socketio=socketio
        )
        video_processors[session_id] = video_processor
        
        active_sessions[session_id] = {
            'client_id': request.sid,
            'type': 'live',
            'camera_index': camera_index
        }
        
        # Start processing in background thread
        thread = threading.Thread(
            target=video_processor.start_live_processing,
            args=(request.sid,)
        )
        thread.daemon = True
        thread.start()
        
        emit('analysis_started', {
            'session_id': session_id,
            'status': 'started'
        })
        
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('start_upload_analysis')
def handle_start_upload_analysis(data):
    """Start analysis of uploaded video"""
    try:
        session_id = data.get('session_id')
        if session_id not in video_processors:
            emit('error', {'message': 'Session not found'})
            return
        
        processor = video_processors[session_id]
        
        # Start processing in background thread
        thread = threading.Thread(
            target=processor.start_upload_processing,
            args=(request.sid,)
        )
        thread.daemon = True
        thread.start()
        
        active_sessions[session_id] = {
            'client_id': request.sid,
            'type': 'upload'
        }
        
        emit('analysis_started', {
            'session_id': session_id,
            'status': 'started'
        })
        
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('stop_analysis')
def handle_stop_analysis(data):
    """Stop video analysis"""
    session_id = data.get('session_id')
    if session_id in video_processors:
        cleanup_session(session_id)
        emit('analysis_stopped', {'session_id': session_id})

@socketio.on('get_tactical_insights')
def handle_tactical_insights(data):
    """Get tactical analysis insights"""
    try:
        session_id = data.get('session_id')
        if session_id not in video_processors:
            emit('error', {'message': 'Session not found'})
            return
        
        processor = video_processors[session_id]
        insights = tactical_analyzer.get_tactical_insights(processor.get_tracking_data())
        
        emit('tactical_insights', insights)
        
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('get_performance_metrics')
def handle_performance_metrics(data):
    """Get performance metrics"""
    try:
        session_id = data.get('session_id')
        if session_id not in video_processors:
            emit('error', {'message': 'Session not found'})
            return
        
        processor = video_processors[session_id]
        metrics = performance_analyzer.get_performance_metrics(processor.get_tracking_data())
        
        emit('performance_metrics', metrics)
        
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('get_advanced_analytics')
def handle_advanced_analytics(data):
    """Get advanced analytics including clustering and anomaly detection"""
    try:
        session_id = data.get('session_id')
        if session_id not in video_processors:
            emit('error', {'message': 'Session not found'})
            return
        
        processor = video_processors[session_id]
        tracking_data = processor.get_tracking_data()
        
        # Perform advanced analytics
        clustering_analysis = advanced_analytics.analyze_player_clustering(tracking_data)
        anomaly_detection = advanced_analytics.detect_anomalies(tracking_data)
        team_sync = advanced_analytics.analyze_team_synchronization(tracking_data)
        heatmap_viz = advanced_analytics.generate_heatmap_visualization(tracking_data)
        
        advanced_results = {
            'clustering': clustering_analysis,
            'anomalies': anomaly_detection,
            'team_synchronization': team_sync,
            'heatmap_visualization': heatmap_viz,
            'timestamp': time.time()
        }
        
        emit('advanced_analytics', advanced_results)
        
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('get_prediction_analysis')
def handle_prediction_analysis(data):
    """Get performance trend predictions"""
    try:
        session_id = data.get('session_id')
        if session_id not in video_processors:
            emit('error', {'message': 'Session not found'})
            return
        
        processor = video_processors[session_id]
        
        # Create historical data from current session
        tracking_data = processor.get_tracking_data()
        historical_data = {
            time.time(): {
                'team_metrics': performance_analyzer.get_performance_metrics(tracking_data)
            }
        }
        
        # Get predictions
        predictions = advanced_analytics.predict_performance_trends(historical_data)
        
        emit('prediction_analysis', predictions)
        
    except Exception as e:
        emit('error', {'message': str(e)})

def cleanup_session(session_id):
    """Clean up a session and its resources"""
    if session_id in video_processors:
        video_processors[session_id].stop()
        del video_processors[session_id]
    
    if session_id in active_sessions:
        del active_sessions[session_id]

if __name__ == '__main__':
    print("Starting Sports Analytics Application...")
    print("Backend server will be available at http://localhost:5000")
    print("Frontend should be running at http://localhost:3000")
    
    # Start the server
    socketio.run(app, host='0.0.0.0', port=5000, debug=True) 
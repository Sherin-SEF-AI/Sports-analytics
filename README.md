# Real-Time Sports Analytics Application

A comprehensive sports analytics platform that provides real-time player and ball tracking, tactical analysis, performance feedback, and automated refereeing assistance for live sports events and uploaded video files.

## Features

### 🎯 Real-Time Player Tracking
- Identify and track all players on the field/court from live video feeds
- Visualize player IDs and movement trajectories as overlays
- Support multiple simultaneous player tracking with minimal latency

### ⚽ Real-Time Ball Tracking
- Accurate ball detection and position tracking
- Real-time ball trajectory visualization
- Ball possession analysis

### 📊 Tactical Analysis Module
- Team formation detection (4-4-2, 4-3-3, etc.)
- Player spacing and distance analysis
- Heatmaps showing player activity zones
- Passing pattern analysis and ball possession statistics

### 📈 Real-Time Performance Feedback
- Individual player metrics (speed, distance, sprints, shot accuracy)
- Team performance indicators (possession %, total shots)
- Real-time dashboard with key statistics

### 🏟️ Automated Refereeing Assistance
- Computer vision algorithms for critical decisions
- Offside detection in football
- Line call assistance for tennis and other sports

### 📹 Video Upload Analysis
- Process pre-recorded sports videos (MP4, AVI)
- Same analysis capabilities as live feeds
- Video player controls with synchronized analytics

## Technical Stack

### Backend
- **Flask**: Web framework with WebSocket support
- **OpenCV**: Computer vision and video processing
- **YOLO (Ultralytics)**: Real-time object detection
- **MediaPipe**: Pose estimation and tracking
- **NumPy/SciPy**: Numerical computations
- **Matplotlib/Seaborn**: Data visualization

### Frontend
- **React**: Modern UI framework
- **Socket.IO**: Real-time communication
- **Canvas API**: Video overlay rendering
- **Chart.js**: Interactive data visualizations

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd sports-analytics
```

2. **Set up Python environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Set up Node.js environment**
```bash
cd frontend
npm install
```

4. **Download YOLO models**
```bash
python scripts/download_models.py
```

## Usage

1. **Start the backend server**
```bash
python app.py
```

2. **Start the frontend development server**
```bash
cd frontend
npm start
```

3. **Access the application**
- Open http://localhost:3000 in your browser
- For live video analysis, connect a camera or video stream
- For uploaded video analysis, use the file upload feature

## Project Structure

```
sports-analytics/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── models/               # AI/ML models
├── utils/                # Utility functions
├── static/               # Static files
├── templates/            # HTML templates
├── frontend/             # React frontend
├── scripts/              # Setup and utility scripts
└── data/                 # Sample data and videos
```

## API Endpoints

- `GET /`: Main application page
- `POST /upload`: Video file upload
- `GET /api/analysis/<video_id>`: Get analysis results
- `WebSocket /socket.io`: Real-time video processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details 
# Sports Analytics Application - Setup Guide

This guide will help you set up and run the Sports Analytics Application on your system.

## Prerequisites

### System Requirements
- **Python 3.8 or higher**
- **OpenCV** (with camera support for live analysis)
- **Webcam or video files** for testing
- **At least 4GB RAM** (8GB recommended for optimal performance)
- **GPU support** (optional, for faster processing)

### Operating System Support
- ✅ Linux (Ubuntu 18.04+, CentOS 7+)
- ✅ Windows 10/11
- ✅ macOS 10.14+

## Installation Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd sports-analytics
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

**Note:** The installation may take several minutes, especially for OpenCV and YOLO dependencies.

### 4. Download AI Models
```bash
# Download YOLO models for object detection
python scripts/download_models.py
```

This will download the YOLOv8n model (~6MB) which is optimized for speed and accuracy.

### 5. Verify Installation
```bash
# Run the test suite
python test_app.py
```

You should see all tests passing with green checkmarks.

## Running the Application

### Start the Backend Server
```bash
python app.py
```

The server will start on `http://localhost:5000`

### Access the Web Interface
Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage Guide

### Live Video Analysis
1. **Connect Camera**: Ensure your webcam is connected and accessible
2. **Start Analysis**: Click "Start Live Analysis" button
3. **View Results**: Real-time player tracking and analytics will appear
4. **Stop Analysis**: Click "Stop Analysis" when finished

### Upload Video Analysis
1. **Select Video**: Click "Choose File" and select a sports video (MP4, AVI, etc.)
2. **Upload**: Click "Upload & Analyze" button
3. **Monitor Progress**: Watch the progress bar as the video is processed
4. **View Results**: Analysis results will appear in the dashboard

### Features Available
- **Real-time Player Tracking**: See player IDs and movement trajectories
- **Ball Tracking**: Track ball position and movement
- **Tactical Analysis**: View team formations and player spacing
- **Performance Metrics**: Individual and team performance statistics
- **Referee Assistance**: Offside detection and foul analysis

## Troubleshooting

### Common Issues

#### 1. Camera Not Detected
```bash
# Check camera availability
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

**Solution**: Ensure camera permissions are granted and no other application is using the camera.

#### 2. YOLO Model Download Fails
```bash
# Manual download
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
mkdir -p models
mv yolov8n.pt models/
```

#### 3. OpenCV Installation Issues
```bash
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-opencv

# For macOS
brew install opencv

# For Windows
pip install opencv-python
```

#### 4. Memory Issues
If you encounter memory errors:
- Reduce video resolution
- Use smaller YOLO model (yolov8n.pt)
- Close other applications
- Increase system swap space

#### 5. Performance Issues
For better performance:
- Use GPU acceleration (if available)
- Reduce frame processing rate
- Use smaller video files for testing

### Error Messages

#### "ModuleNotFoundError: No module named 'cv2'"
```bash
pip install opencv-python
```

#### "ModuleNotFoundError: No module named 'ultralytics'"
```bash
pip install ultralytics
```

#### "Permission denied" for camera
```bash
# Linux
sudo usermod -a -G video $USER
# Then log out and log back in
```

## Configuration

### Environment Variables
Create a `.env` file in the project root:
```env
FLASK_ENV=development
FLASK_DEBUG=1
CAMERA_INDEX=0
MODEL_PATH=models/yolov8n.pt
```

### Custom Settings
Edit `app.py` to modify:
- Camera index for live analysis
- Video processing parameters
- WebSocket settings
- Upload file size limits

## Development

### Project Structure
```
sports-analytics/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── utils/                 # Core analysis modules
│   ├── video_processor.py
│   ├── player_tracker.py
│   ├── ball_tracker.py
│   ├── tactical_analyzer.py
│   ├── performance_analyzer.py
│   └── referee_assistant.py
├── templates/             # HTML templates
├── static/                # Static files
├── uploads/               # Uploaded videos
├── models/                # AI models
├── scripts/               # Setup scripts
└── data/                  # Sample data
```

### Adding New Features
1. Create new module in `utils/`
2. Import in `app.py`
3. Add WebSocket events for real-time updates
4. Update frontend to display new data

## Performance Optimization

### For Production Use
1. **Use Production WSGI Server**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
   ```

2. **Enable GPU Acceleration**:
   ```bash
   pip install torch torchvision
   ```

3. **Optimize Model Loading**:
   - Pre-load models on startup
   - Use model caching
   - Implement lazy loading

4. **Database Integration**:
   - Add PostgreSQL for data persistence
   - Implement caching with Redis
   - Add user authentication

## Support

### Getting Help
1. Check the troubleshooting section above
2. Review error logs in the console
3. Test with sample videos first
4. Ensure all dependencies are installed

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **YOLO**: Real-time object detection
- **OpenCV**: Computer vision library
- **Flask**: Web framework
- **Socket.IO**: Real-time communication
- **MediaPipe**: Pose estimation (optional)

---

**Happy Analyzing!** 🏈⚽🏀 
# Football Analysis Project

## Project Overview
This project is a comprehensive football analytics system that uses computer vision and machine learning to analyze football matches. It provides real-time tracking of players, ball possession analysis, and tactical insights through various AI-powered modules.

## Key Features
- Player detection and tracking using YOLO
- Team assignment based on jersey colors using K-means clustering
- Ball possession analysis
- Camera movement tracking via optical flow
- Perspective transformation for accurate measurements
- Speed and distance calculations
- Tactical view generation

## Models Used
### 1. YOLO Models
- `football_player_detector.pt`: Custom-trained YOLOv8 model for player detection
- `football_keypoint_detector.pt`: Specialized model for court keypoint detection
- Base Architecture: YOLOv8 (ultralytics)
- Training Dataset: Custom annotated football match footage

### 2. Computer Vision Models
- K-means clustering for jersey color segmentation
- Lucas-Kanade optical flow for camera motion estimation
- Homography transformation for perspective correction

## Installation & Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/football-analysis.git
cd football-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models:
- Download [Trained YOLOv5 Model](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)
- Place in `models/` directory

## Project Structure
```
├── ball_acquisition/           # Ball tracking and possession analysis
├── camera_movement_estimator/  # Optical flow implementation
├── court_keypoint_detector/    # Field landmark detection
├── models/                     # Pre-trained model storage
├── team_assigner/             # Jersey color analysis
└── trackers/                  # Player tracking implementation
```

## Usage
1. Place your input video in the `input_videos/` directory
2. Run the main script:
```bash
python main.py --input input_videos/your_video.mp4 --output output_videos/result.mp4
```

## Components
### 1. Player Detection & Tracking
- Uses YOLOv8 for player detection
- Implementation: `trackers/tracker.py`
- Features: Multi-object tracking, occlusion handling

### 2. Team Assignment
- Uses K-means clustering for jersey color analysis
- Implementation: `team_assigner/team_assigner.py`
- Features: Automatic team identification

### 3. Ball Possession Analysis
- Tracks ball movement and possession
- Implementation: `ball_acquisition/ball_acquisition_detector.py`
- Features: Possession statistics, pass detection

### 4. Speed & Distance Calculation
- Uses perspective transformation for accurate measurements
- Implementation: `speed_and_distance_estimator/speed_and_distance_estimator.py`
- Features: Player speed tracking, distance covered

## Sample Data
- [Sample Input Video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)
- Output Example:
![Screenshot](output_videos/screenshot.png)

## Requirements
- Python 3.x
- ultralytics>=8.0.0
- supervision>=0.3.0
- opencv-python>=4.7.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- pandas>=2.0.0

## Training
Custom training notebooks are available in `training/`:
- `football_player_detection_training.ipynb`
- `football-court-keypoint.ipynb`

## License
This project is licensed under the MIT License - see the LICENSE file for details.
"""
Configuration file for RFDETR-SEG model integration.
Update these values with your actual Roboflow credentials.
"""

# RFDETR-SEG Model Configuration
RFDETR_CONFIG = {
    "api_key": "2t6xilh0yCbvKGshqtgh",  # Replace with your Roboflow API key
    "workspace": "moumitas",       # Replace with your Roboflow workspace
    "project": "football-rfdetr-seg-jtcdc",  # Replace with your project name
    "version": 1,                        # Replace with your model version
    "confidence_threshold": 0.5         # Confidence threshold for detections
}

# Class mappings for RFDETR-SEG
CLASS_MAPPINGS = {
    0: "ball",
    1: "goalkeeper", 
    2: "player",
    3: "referee"
}

# Detection settings
DETECTION_SETTINGS = {
    "batch_size": 20,
    "confidence_threshold": 0.1,  # For tracking
    "possession_threshold": 15    # For ball acquisition detection
}

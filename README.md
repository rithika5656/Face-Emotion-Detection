# Face Emotion Detection

A simple webcam-based face emotion detection system using Python, OpenCV, and CNN.

## Features
- Real-time face detection using OpenCV
- Emotion classification: Happy, Sad, Angry
- Simple CNN model architecture

## Requirements
- Python 3.8+
- OpenCV
- TensorFlow
- NumPy

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train the model (optional)
```bash
python train_model.py
```

### Run emotion detection
```bash
python main.py
```

Press 'q' to quit the application.

## Project Structure
```
├── main.py              # Main application with webcam
├── model.py             # CNN model architecture
├── train_model.py       # Model training script
├── requirements.txt     # Dependencies
└── README.md           # Documentation
```

## How it works
1. Captures video from webcam
2. Detects faces using Haar Cascade classifier
3. Preprocesses face images (grayscale, resize to 48x48)
4. Predicts emotion using trained CNN model
5. Displays result on screen

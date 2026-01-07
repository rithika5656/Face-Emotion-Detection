"""
Face Emotion Detection - Main Application
Real-time webcam-based emotion detection using OpenCV and CNN
Detects: Happy, Sad, Angry
"""

import cv2
import numpy as np
from model import EMOTIONS, load_trained_model, create_model

def preprocess_face(face_img):
    """Preprocess face image for model prediction"""
    # Convert to grayscale if needed
    if len(face_img.shape) == 3:
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_img
    
    # Resize to 48x48
    face_resized = cv2.resize(face_gray, (48, 48))
    
    # Normalize pixel values
    face_normalized = face_resized / 255.0
    
    # Reshape for model input (batch_size, height, width, channels)
    face_input = face_normalized.reshape(1, 48, 48, 1)
    
    return face_input

def get_emotion_color(emotion):
    """Get color for emotion label"""
    colors = {
        'Happy': (0, 255, 0),    # Green
        'Sad': (255, 0, 0),      # Blue
        'Angry': (0, 0, 255)     # Red
    }
    return colors.get(emotion, (255, 255, 255))

def main():
    """Main function to run webcam emotion detection"""
    print("=" * 50)
    print("Face Emotion Detection")
    print("=" * 50)
    print("Emotions: Happy, Sad, Angry")
    print("Press 'q' to quit")
    print("=" * 50)
    
    # Load face detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Load or create emotion model
    model = load_trained_model('emotion_model.h5')
    if model is None:
        print("\nCreating new model for demonstration...")
        model = create_model()
        print("Note: Train the model for accurate predictions!")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\nWebcam started successfully!")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48)
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Preprocess face for prediction
            face_input = preprocess_face(face_roi)
            
            # Predict emotion
            predictions = model.predict(face_input, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            emotion = EMOTIONS[emotion_idx]
            confidence = predictions[0][emotion_idx] * 100
            
            # Get color for emotion
            color = get_emotion_color(emotion)
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Display emotion label
            label = f"{emotion}: {confidence:.1f}%"
            cv2.putText(
                frame, label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2
            )
        
        # Display instructions
        cv2.putText(
            frame, "Press 'q' to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )
        
        # Show frame
        cv2.imshow('Face Emotion Detection', frame)
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nApplication closed.")

if __name__ == "__main__":
    main()

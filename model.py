"""
CNN Model Architecture for Face Emotion Detection
Detects: Happy, Sad, Angry
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

# Emotion labels
EMOTIONS = ['Angry', 'Happy', 'Sad']

def create_model():
    """
    Create a simple CNN model for emotion detection
    Input: 48x48 grayscale images
    Output: 3 classes (Angry, Happy, Sad)
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # 3 emotions
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_trained_model(model_path='emotion_model.h5'):
    """Load a pre-trained model"""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except:
        print("No trained model found. Please train the model first.")
        return None

if __name__ == "__main__":
    model = create_model()
    model.summary()

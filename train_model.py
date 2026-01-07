"""
Training Script for Face Emotion Detection Model
Creates synthetic data for demonstration purposes
"""

import numpy as np
from tensorflow.keras.utils import to_categorical
from model import create_model, EMOTIONS

def generate_synthetic_data(num_samples=1000):
    """
    Generate synthetic training data for demonstration
    In real use, replace with actual emotion dataset (FER2013, etc.)
    """
    print("Generating synthetic training data...")
    
    # Generate random images (48x48 grayscale)
    X = np.random.rand(num_samples, 48, 48, 1).astype('float32')
    
    # Generate random labels (0: Angry, 1: Happy, 2: Sad)
    y = np.random.randint(0, 3, num_samples)
    y = to_categorical(y, num_classes=3)
    
    return X, y

def train_model():
    """Train the emotion detection model"""
    
    # Create model
    print("Creating CNN model...")
    model = create_model()
    
    # Generate synthetic data (replace with real dataset for production)
    X_train, y_train = generate_synthetic_data(num_samples=1000)
    X_val, y_val = generate_synthetic_data(num_samples=200)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Emotions: {EMOTIONS}")
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=10,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Save the model
    model.save('emotion_model.h5')
    print("\nModel saved as 'emotion_model.h5'")
    
    return model, history

if __name__ == "__main__":
    print("=" * 50)
    print("Face Emotion Detection - Model Training")
    print("=" * 50)
    print("\nNote: This uses synthetic data for demonstration.")
    print("For production, use a real dataset like FER2013.\n")
    
    model, history = train_model()
    
    print("\nTraining complete!")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

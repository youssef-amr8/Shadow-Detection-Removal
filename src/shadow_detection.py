import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt

from preprocessing import load_dataset, IMG_SIZE
from model import build_unet

def train_shadow_detection(data_path, epochs=50, batch_size=8):
    """
    Train U-Net for shadow detection
    """
    print("=" * 50)
    print("SHADOW DETECTION TRAINING")
    print("=" * 50)
    
    # 1. Load dataset
    print("\n[1/5] Loading dataset...")
    train_path = os.path.join(data_path, "train")
    X_train, y_train = load_dataset(train_path)
    print(f"Loaded {len(X_train)} training samples")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # 2. Split into train/validation (80/20)
    print("\n[2/5] Splitting dataset...")
    split_idx = int(0.8 * len(X_train))
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # 3. Build model
    print("\n[3/5] Building U-Net model...")
    model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # 4. Setup callbacks
    print("\n[4/5] Setting up training callbacks...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Model checkpoint - save best model
    checkpoint = callbacks.ModelCheckpoint(
        filepath='models/shadow_detection.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    # Early stopping - stop if no improvement
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    callbacks_list = [checkpoint, early_stop]
    
    # 5. Train model
    print("\n[5/5] Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1
    )
    
    print("\n✅ Shadow detection model saved to 'models/shadow_detection.h5'")
    
    return model, history

def predict_shadow_mask(model, image):
    # Add batch dimension if needed
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # Predict
    shadow_prob = model.predict(image, verbose=0)
    
    # Convert to binary mask (threshold at 0.5)
    binary_mask = (shadow_prob > 0.5).astype(np.float32)
    
    # Remove batch dimension if input was single image
    if shadow_prob.shape[0] == 1:
        shadow_prob = shadow_prob[0]
        binary_mask = binary_mask[0]
    
    return shadow_prob, binary_mask

if __name__ == "__main__":
    # Train the model when script is run directly
    model, history = train_shadow_detection(
        data_path='data/ISTD_Dataset',
        epochs=50,
        batch_size=8
    )
import tensorflow as tf
import os

MODEL_PATH = "brain-tumor-model.h5"

if os.path.exists(MODEL_PATH):
    print(f"Model file exists: {MODEL_PATH}")
    print(f"File size: {os.path.getsize(MODEL_PATH)} bytes")
    
    try:
        print("\n1. Trying with compile=False...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"SUCCESS! Input shape: {model.input_shape}")
    except Exception as e:
        print(f"FAILED: {e}")
        
    try:
        print("\n2. Trying with safe_mode=False...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        print(f"SUCCESS! Input shape: {model.input_shape}")
    except Exception as e:
        print(f"FAILED: {e}")
        
else:
    print(f"Model file NOT found: {MODEL_PATH}")

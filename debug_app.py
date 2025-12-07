import torch
import torch.nn as nn
from torchvision import models
import os
import json

# Configuration
MODEL_PATH = "best_model_state.pth"
CLASSES_PATH = "classes.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")

try:
    # 1. Load Classes
    print("Loading classes...")
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, 'r') as f:
            classes = json.load(f)
        print(f"Classes loaded: {classes}")
    else:
        print("Classes file not found.")
        classes = ["C1", "C2", "C3", "C4"] # Dummy

    num_classes = len(classes)

    # 2. Build Model
    print("Building model...")
    model = models.mobilenet_v2(weights=None)
    model.last_channel = 1280
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    # 3. Load State Dict
    print(f"Loading state dict from {MODEL_PATH}...")
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device)
        
        # Handle DataParallel
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # Load
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        
        # 4. Dummy Inference
        print("Running dummy inference...")
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print("Success!")
        
    else:
        print("Model file not found!")

except Exception as e:
    print("\nXXX ERROR OCCURRED XXX")
    print(e)
    import traceback
    traceback.print_exc()

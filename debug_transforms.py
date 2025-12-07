import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np

# Configuration
print("Testing PIL and Transforms...")
try:
    # Create dummy image
    img = Image.fromarray(np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8))
    
    # Define transforms (SAME as app.py)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Apply
    out = transform(img)
    print(f"Transform output shape: {out.shape}")
    print("Transforms OK!")
    
except Exception as e:
    print("\nXXX TRANSFORM ERROR XXX")
    print(e)
    import traceback
    traceback.print_exc()

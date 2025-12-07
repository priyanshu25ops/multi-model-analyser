import requests
from PIL import Image
import io
import numpy as np

# Create a dummy MRI image
img = Image.fromarray(np.random.randint(0, 255, (300, 300), dtype=np.uint8), mode='L')  # Grayscale
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
img_byte_arr.seek(0)

# Test MRI endpoint
print("Testing /predict_mri...")
url_mri = 'http://127.0.0.1:8000/predict_mri'
files_mri = {'file': ('test_mri.png', img_byte_arr, 'image/png')}

try:
    response = requests.post(url_mri, files=files_mri)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")

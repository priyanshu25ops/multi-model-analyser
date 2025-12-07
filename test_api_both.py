import requests
from PIL import Image
import io
import numpy as np

# Create a dummy X-ray image
img = Image.fromarray(np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8))
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='JPEG')
img_byte_arr.seek(0)

# Test X-Ray endpoint
print("Testing /predict (X-Ray)...")
url = 'http://127.0.0.1:8000/predict'
files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}

try:
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")

# Reset for MRI test
img_byte_arr.seek(0)

# Test MRI endpoint
print("\n\nTesting /predict_mri (MRI)...")
url_mri = 'http://127.0.0.1:8000/predict_mri'
files_mri = {'file': ('test_mri.jpg', img_byte_arr, 'image/jpeg')}

try:
    response = requests.post(url_mri, files=files_mri)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")

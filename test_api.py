import requests
from PIL import Image
import io
import numpy as np

# Create a random image
img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='JPEG')
img_byte_arr.seek(0)

url = 'http://127.0.0.1:8000/predict'
files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}

try:
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")

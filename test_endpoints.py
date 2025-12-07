import requests
import os

# Test MRI endpoint
print("Testing MRI endpoint...")
if os.path.exists("brain-tumor-model.h5"):
    # Create dummy MRI test
    from PIL import Image
    import numpy as np
    import io
    
    img = Image.fromarray(np.random.randint(0, 255, (300, 300), dtype=np.uint8), mode='L')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    try:
        response = requests.post('http://127.0.0.1:8000/predict_mri', files={'file': ('test.png', img_bytes, 'image/png')})
        print(f"MRI Status: {response.status_code}")
        print(f"MRI Response: {response.text[:200]}")
    except Exception as e:
        print(f"MRI Error: {e}")
else:
    print("brain-tumor-model.h5 not found!")

# Test Medical Report endpoint  
print("\n\nTesting Medical Report endpoint...")
if os.path.exists(".env"):
    # Check if sample PDF exists
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    if pdf_files:
        test_pdf = pdf_files[0]
        print(f"Using PDF: {test_pdf}")
        try:
            with open(test_pdf, 'rb') as f:
                response = requests.post('http://127.0.0.1:8000/analyze_report', files={'file': (test_pdf, f, 'application/pdf')})
            print(f"Report Status: {response.status_code}")
            print(f"Report Response: {response.text[:200]}")
        except Exception as e:
            print(f"Report Error: {e}")
    else:
        print("No PDF files found to test. Create a dummy one...")
        # Just test if endpoint exists
        try:
            response = requests.get('http://127.0.0.1:8000/')
            print(f"Server is running: {response.status_code}")
        except:
            print("Server not responding!")
else:
    print(".env not found - Groq API key missing!")

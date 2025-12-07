import requests
import os
import sys

def test_backend():
    print("--- Testing Backend Endpoints ---")
    
    # Test 1: MRI Model Loading & Prediction
    print("1. Testing MRI Prediction...")
    if os.path.exists("brain-tumor-model.h5"):
        from PIL import Image
        import numpy as np
        import io
        
        # Create dummy grayscale image (300x300)
        img = Image.fromarray(np.random.randint(0, 255, (300, 300), dtype=np.uint8), mode='L')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        try:
            response = requests.post('http://127.0.0.1:8000/predict_mri', files={'file': ('test.png', img_bytes, 'image/png')})
            if response.status_code == 200:
                print("   [PASS] MRI Prediction SUCCESS")
                print(f"   Response: {response.json()}")
            else:
                print(f"   [FAIL] MRI Prediction FAILED: {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"   [FAIL] MRI Connection Error: {e}")
    else:
        print("   [WARN] MRI Model file missing")

    # Test 2: Medical Report Analysis
    print("\n2. Testing Medical Report Analysis...")
    if os.path.exists(".env"):
        # Create dummy PDF
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Patient shows signs of normal brain function. No tumors detected.")
        pdf_bytes = doc.tobytes()
        
        try:
            response = requests.post('http://127.0.0.1:8000/analyze_report', files={'file': ('test_report.pdf', pdf_bytes, 'application/pdf')})
            if response.status_code == 200:
                print("   [PASS] Report Analysis SUCCESS")
                print(f"   Response Preview: {str(response.json())[:100]}...")
            else:
                print(f"   [FAIL] Report Analysis FAILED: {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"   [FAIL] Report Connection Error: {e}")
    else:
        print("   [WARN] .env file missing (Groq API key needed)")

def verify_frontend_fix():
    print("\n--- Verifying Frontend Fix (index.html) ---")
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if barsContainer is defined BEFORE it is used
        display_results_start = content.find('function displayResults(data)')
        bars_def = content.find("const barsContainer = document.getElementById('bars');", display_results_start)
        bars_usage = content.find("barsContainer.innerHTML = '';", display_results_start)
        
        if bars_def != -1 and bars_usage != -1 and bars_def < bars_usage:
            print("   [PASS] Fix Verified: 'barsContainer' is defined before use.")
        else:
            print("   [FAIL] Fix FAILED: 'barsContainer' might be used before definition!")
            print(f"   Def pos: {bars_def}, Usage pos: {bars_usage}")
            
        # Check if broken script tag is gone
        if '/static/fix.js' not in content:
             print("   [PASS] Cleaned: Broken script tag removed.")
        else:
             print("   [WARN] Warning: Broken script tag still present.")
             
    except Exception as e:
        print(f"   ❌ Error reading index.html: {e}")

if __name__ == "__main__":
    try:
        test_backend()
        verify_frontend_fix()
    except Exception as e:
        print(f"Script Error: {e}")

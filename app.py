import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
import io
import os

app = FastAPI()

# Configuration
MODEL_PATH = "best_model_state.pth"
CLASSES_PATH = "classes.json"


# Validates numpy for TensorFlow
import numpy as np
import tensorflow as tf

# Medical Report Analysis imports
from dotenv import load_dotenv
import fitz  # PyMuPDF
from groq import Groq

# Load environment variables
load_dotenv()

# MRI Configuration
MRI_MODEL_PATH = "brain-tumor-model.h5"
mri_model = None
mri_classes = ["glioma", "meningioma", "no_tumor", "pituitary"]
mri_input_shape = (None, 300, 300, 1) # Default fallback - grayscale 300x300

# X-Ray Configuration
model = None
classes = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Update load_resources to load MRI model too
def load_resources():
    global model, classes, mri_model, mri_input_shape
    
    # --- X-RAY ---
    # Load Classes
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, 'r') as f:
            classes = json.load(f)
    else:
        classes = ["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]
        print(f"Warning: {CLASSES_PATH} not found. Using default classes.")

    num_classes = len(classes)

    # Load Model structure (X-Ray)
    model = models.mobilenet_v2(weights=None) 
    model.last_channel = 1280
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    # Load State Dict (X-Ray)
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print("X-Ray Model loaded successfully.")
        except Exception as e:
            print(f"Error loading X-Ray model: {e}")
    else:
        print(f"Error: {MODEL_PATH} not found.")

    # --- MRI ---
    if os.path.exists(MRI_MODEL_PATH):
        try:
            print("Loading MRI Model...")
            # Try loading with safe_mode=False to bypass strict deserialization
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mri_model = tf.keras.models.load_model(
                    MRI_MODEL_PATH, 
                    compile=False,
                    safe_mode=False
                )
            mri_input_shape = mri_model.input_shape
            print(f"MRI Model loaded! Expects shape: {mri_input_shape}")
        except Exception as e:
            print(f"Error loading MRI model: {e}")
            print("Attempting alternative loading method...")
            try:
                # Fallback: load with custom objects
                mri_model = tf.keras.models.load_model(MRI_MODEL_PATH, compile=False)
                mri_input_shape = mri_model.input_shape if mri_model else (None, 150, 150, 3)
                print(f"MRI Model loaded with fallback! Shape: {mri_input_shape}")
            except:
                print("MRI model loading failed completely. MRI endpoint will be unavailable.")
    else:
        print(f"Warning: {MRI_MODEL_PATH} not found.")

# Initialize on startup
load_resources()

# Image Transforms (X-Ray)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r") as f:
        return f.read()

@app.get("/classes")
async def get_classes():
    return {"xray_classes": classes, "mri_classes": mri_classes}

@app.post("/predict")
async def predict_xray(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "X-Ray Model not loaded"})
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = classes[predicted_idx.item()]
        
        class_probs = {cls_name: float(prob) for cls_name, prob in zip(classes, probabilities.tolist())}
        
        return {
            "model_type": "X-Ray",
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "probabilities": class_probs
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict_mri")
async def predict_mri(file: UploadFile = File(...)):
    if mri_model is None:
        return JSONResponse(status_code=500, content={"error": "MRI Model not loaded"})

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Determine target shape from model (usually (None, H, W, C))
        # mri_input_shape is like (None, 150, 150, 3)
        _, H, W, C = mri_input_shape
        
        # Preprocessing
        if C == 1:
            if image.mode != "L": image = image.convert("L")
        else:
            if image.mode != "RGB": image = image.convert("RGB")

        # Resize
        image = image.resize((W, H))
        
        # Normalize
        img_array = np.array(image).astype("float32") / 255.0
        
        # Expand dims
        if C == 1:
            img_array = np.expand_dims(img_array, axis=-1)
        
        img_array = np.expand_dims(img_array, axis=0) # Batch dim
        
        # Inference
        preds = mri_model.predict(img_array)[0]
        
        pred_idx = int(np.argmax(preds))
        predicted_class = mri_classes[pred_idx]
        confidence = float(preds[pred_idx])
        
        class_probs = {cls_name: float(prob) for cls_name, prob in zip(mri_classes, preds)}
        
        return {
            "model_type": "MRI",
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": class_probs
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/analyze_report")
async def analyze_report(
    file: UploadFile = File(...), 
    query: str = Form(""),
    custom_prompt: str = Form(None)
):
    """
    Analyze medical report PDF using Groq API.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return JSONResponse(status_code=500, content={"error": "Groq API key not configured"})
    
    try:
        # Read PDF file
        pdf_bytes = await file.read()
        
        # Extract text from PDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        report_text = "\n".join(page.get_text() for page in doc)
        doc.close()
        
        if not report_text.strip():
            return JSONResponse(status_code=400, content={
                "error": "No text text found in PDF! This appears to be a scanned image. Please upload a standard digital PDF with selectable text."
            })
        
        # Prepare prompt
        user_query = query.strip() if query else "Please analyze this medical report and explain the findings in simple terms."
        
        # Call Groq API
        client = Groq(api_key=groq_api_key)
        
        default_system_prompt = """You are an experienced and empathetic Doctor. Analyze the provided medical report detailedly.

Please structure your response with the following sections using simple headers and bullet points. DO NOT use bolding or asterisks (like **text**). Use plain text.

1. Clinical Analysis
- detailed breakdown of the key findings.
- Explain any abnormal values or terms in simple language.

2. Severity Assessment
- Clearly state if there are any severe or critical issues mentioned.
- Rate the overall concern level (Low/Medium/High).

3. Recommendations & Next Steps
- Suggest immediate actions (e.g., rest, diet changes, hydration).
- Mention if any specific prescription or medication types might be typically considered (Note: Always add a disclaimer that this is not a final prescription).

4. Consultation Advice
- Explicitly state whether they should consult a specialist (e.g., Cardiologist, GP) and how urgently.

Disclaimer: This analysis is AI-generated for informational purposes and does not replace professional medical advice."""

        # Use custom prompt if provided
        final_system_prompt = custom_prompt if (custom_prompt and custom_prompt.strip()) else default_system_prompt

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # or llama-4-maverick
            messages=[
                {
                    "role": "system",
                    "content": final_system_prompt
                },
                {
                    "role": "user",
                    "content": f"Patient Query: {user_query}\n\nReport Contents:\n{report_text[:8000]}"
                }
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        analysis = completion.choices[0].message.content
        
        return {
            "model_type": "Medical Report",
            "analysis": analysis,
            "query": user_query
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


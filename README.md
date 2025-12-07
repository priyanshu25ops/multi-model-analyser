

# 🧠 Multi-Model Analyser

[](https://www.python.org/)
[](http://flask.pocoo.org/)
[](https://www.tensorflow.org/)
[](https://groq.com/)

-----

## 📝 Overview

The **Multi-Model Analyser** is a lightweight Flask web application that performs **diagnostic analysis on medical images** (e.g., Brain MRI scans). It features a robust architecture that compares predictions from **multiple deep learning models** (TensorFlow/Keras and PyTorch) for enhanced reliability.

### ⚡️ Advanced Feature: AI Contextual Analysis (Groq)

This project is enhanced with the **Groq API** to provide **lightning-fast, human-readable contextual analysis** on top of the traditional deep learning classification.

A dedicated custom path, such as `/api/groq_analysis`, is used to send the raw model outputs and image context to a high-speed LLM (like Llama 3 or Mixtral). This allows the application to generate a comprehensive, natural-language explanation of the findings, adding significant value beyond simple labels.

-----

## ✨ Core Features

  * **Multi-Framework Support:** Handles model inference from both **TensorFlow/Keras** (`.h5` files) and **PyTorch** (`.pth` files).
  * **Hybrid Analysis:** Combines traditional ML classification with cutting-edge **Groq LLM inference** for complete diagnostic reporting.
  * **RESTful API Endpoints:** Exposes routes for individual model predictions, the combined classification result, and the specialized **Groq analysis path**.
  * **Simple Web UI:** An HTML interface allows users to upload images and view both the model scores and the final AI-generated summary.

-----

## 🚀 Getting Started

### 1\. Prerequisites

  * Python 3.x
  * Model and configuration files (`.h5`, `.pth`, `classes.json`)
  * **A Groq API Key** (Essential for the LLM analysis feature).

### 2\. Groq API Key Configuration

For security, the Groq API key is accessed via an environment variable.

1.  **Get Your Key:** Obtain your key from the [Groq Console](https://console.groq.com/).

2.  **Set Environment Variable:**

    ```bash
    # Set the key in your terminal before running the app:
    export GROQ_API_KEY="gsk_YourSecretKeyHere"
    ```

### 3\. Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/priyanshu25ops/multi-model-analyser
    cd multi-model-analyser
    ```

2.  **Install Dependencies:**

    The project requires the standard ML frameworks plus the new `groq` library.

    ```bash
    pip install Flask tensorflow torch torchvision numpy Pillow groq
    ```

### 4\. Running the Application

Start the Flask development server:

```bash
python app.py
```

Access the application in your web browser at: **[http://127.0.0.1:5000](https://www.google.com/search?q=http://127.0.0.1:5000)**

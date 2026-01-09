import os
import re
import base64
import io
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

app = Flask(__name__)
CORS(app)

# --- 1. Model Configuration ---
# Your Hugging Face Repo ID
MODEL_PATH = "harun-767/dog-breed-classifier"

# IMPORTANT: Set this to None since your files are in the root
SUBFOLDER = None 

print(f"Loading model from {MODEL_PATH}...")

try:
    # Load the Processor (handles resizing/normalization)
    # We remove the 'subfolder' argument entirely if it's None
    if SUBFOLDER:
        processor = ViTImageProcessor.from_pretrained(MODEL_PATH, subfolder=SUBFOLDER)
        model = ViTForImageClassification.from_pretrained(MODEL_PATH, subfolder=SUBFOLDER)
    else:
        # Load directly from root
        processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
        model = ViTForImageClassification.from_pretrained(MODEL_PATH)
        
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Critical Error loading model: {e}")
    model = None

@app.route("/", methods=["GET"])
def home():
    return "🐶 Dog Breed AI is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server."}), 500

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # --- 2. Process Base64 Image ---
        b64_string = data["image"]
        
        # Clean up the string (remove "data:image/png;base64," prefix if present)
        b64_string = re.sub(r"^data:image/.+;base64,", "", b64_string)

        # Fix potential padding errors in the Base64 string
        missing_padding = len(b64_string) % 4
        if missing_padding:
            b64_string += "=" * (4 - missing_padding)

        # Decode and convert to RGB
        image_bytes = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # --- 3. AI Prediction ---
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)

        # Calculate probabilities
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]

        # Get Top 3 results
        top_k = torch.topk(probs, 3)
        results = []
        for score, idx in zip(top_k.values, top_k.indices):
            results.append({
                "label": model.config.id2label[idx.item()],
                "confidence": round(score.item(), 4)
            })

        return jsonify({
            "status": "success",
            "predictions": results
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
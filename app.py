import os
import re
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Hugging Face Inference API ---
MODEL_ID = "harun-767/dog-breed-classifier"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HF_TOKEN = os.getenv("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

@app.route("/", methods=["GET"])
def home():
    return "🐶 Dog Breed AI (Remote Inference via Hugging Face) is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if not HF_TOKEN:
        return jsonify({"error": "HF_TOKEN not set"}), 500

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # --- Clean Base64 ---
        b64_string = re.sub(r"^data:image/.+;base64,", "", data["image"])
        missing_padding = len(b64_string) % 4
        if missing_padding:
            b64_string += "=" * (4 - missing_padding)

        image_bytes = base64.b64decode(b64_string)

        # --- Send to Hugging Face Inference API ---
        response = requests.post(
            HF_API_URL,
            headers=HEADERS,
            data=image_bytes,
            timeout=60
        )

        if response.status_code != 200:
            return jsonify({
                "error": "Hugging Face inference failed",
                "details": response.text
            }), response.status_code

        hf_result = response.json()

        # Normalize output (HF returns list of {label, score})
        predictions = [
            {
                "label": pred["label"],
                "confidence": round(pred["score"], 4)
            }
            for pred in hf_result
        ]

        return jsonify({
            "status": "success",
            "predictions": predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

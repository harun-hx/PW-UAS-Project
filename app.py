import os
import re
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# -----------------------------
# App setup
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Hugging Face config
# -----------------------------
MODEL_ID = "harun-767/dog-breed-classifier"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HF_TOKEN = os.getenv("HF_TOKEN")

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "🐶 Dog Breed AI (Remote Inference) is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    # --- 1. Check HF token ---
    if not HF_TOKEN:
        return jsonify({"error": "HF_TOKEN not set on server"}), 500

    # --- 2. Validate request ---
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Request JSON must contain 'image' field"}), 400

    try:
        # --- 3. Clean & decode base64 image ---
        b64_string = data["image"]

        # Remove data URL prefix if present
        b64_string = re.sub(r"^data:image/.+;base64,", "", b64_string)

        # Fix base64 padding
        b64_string += "=" * (-len(b64_string) % 4)

        image_bytes = base64.b64decode(b64_string)

        # --- 4. Call Hugging Face Inference API ---
        response = requests.post(
            HF_API_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/octet-stream"
            },
            data=image_bytes,
            timeout=60
        )

        # --- 5. Parse HF response ---
        try:
            hf_result = response.json()
        except Exception:
            return jsonify({
                "error": "Invalid response from Hugging Face",
                "raw": response.text
            }), 502

        # HF error (model loading, auth, etc.)
        if isinstance(hf_result, dict) and "error" in hf_result:
            return jsonify({
                "error": "Hugging Face inference error",
                "details": hf_result["error"]
            }), 503

        # Expect list of predictions
        if not isinstance(hf_result, list):
            return jsonify({
                "error": "Unexpected HF response format",
                "raw": hf_result
            }), 500

        # --- 6. Normalize, sort, and take TOP-5 ---
        predictions = [
            {
                "label": pred.get("label"),
                "confidence": float(pred.get("score", 0))
            }
            for pred in hf_result
        ]

        # Sort by confidence (descending)
        predictions.sort(key=lambda x: x["confidence"], reverse=True)

        # Take top 5
        top5 = predictions[:5]

        # Round confidence for clean output
        for p in top5:
            p["confidence"] = round(p["confidence"], 4)

        return jsonify({
            "status": "success",
            "top1": top5[0] if top5 else None,
            "top5": top5
        })

    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500


# -----------------------------
# Local run (Railway uses gunicorn)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

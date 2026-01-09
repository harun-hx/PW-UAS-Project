import os
import re
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_ID = "harun-767/dog-breed-classifier"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HF_TOKEN = os.getenv("HF_TOKEN")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "🐶 Dog Breed AI (Remote Inference) is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not HF_TOKEN:
        return jsonify({"error": "HF_TOKEN not set on server"}), 500

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "JSON must contain 'image'"}), 400

    try:
        # ---- Clean base64 ----
        b64_string = data["image"]
        b64_string = re.sub(r"^data:image/.+;base64,", "", b64_string)
        b64_string += "=" * (-len(b64_string) % 4)

        # ---- HF expects base64 inside JSON inputs ----
        payload = {
            "inputs": b64_string
        }

        response = requests.post(
            HF_API_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
                "X-Wait-For-Model": "true"
            },
            json=payload,
            timeout=120
        )

        # ---- HARD DEBUG (important) ----
        if response.text.strip() == "":
            return jsonify({
                "error": "Empty response from Hugging Face",
                "status_code": response.status_code
            }), 502

        # Try parsing JSON safely
        try:
            hf_result = response.json()
        except Exception:
            return jsonify({
                "error": "Non-JSON response from Hugging Face",
                "raw": response.text[:500],
                "status_code": response.status_code
            }), 502

        # HF error payload
        if isinstance(hf_result, dict) and "error" in hf_result:
            return jsonify({
                "error": "Hugging Face inference error",
                "details": hf_result["error"]
            }), 503

        if not isinstance(hf_result, list):
            return jsonify({
                "error": "Unexpected HF response format",
                "raw": hf_result
            }), 500

        # ---- Sort & Top-5 ----
        predictions = sorted(
            [
                {
                    "label": p.get("label"),
                    "confidence": float(p.get("score", 0))
                }
                for p in hf_result
            ],
            key=lambda x: x["confidence"],
            reverse=True
        )[:5]

        for p in predictions:
            p["confidence"] = round(p["confidence"], 4)

        return jsonify({
            "status": "success",
            "top1": predictions[0],
            "top5": predictions
        })

    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

import os
import base64
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

import tensorflow as tf
MODEL_PATH = os.path.join(os.path.dirname(__file__), "paudhamitra_model.keras")
_model = None

def get_model():
    global _model
    if _model is None:
        print(f"Loading TensorFlow model from {MODEL_PATH} ...")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    return _model

CLASS_NAMES = [
    "Aloevera_healthyLeaf",
    "Aloevera_sunstressLeaf",
    "Cactus_healthyCactus",
    "Cactus_healthyCactus2",
    "Dracaena_healthyLeaf",
    "Dracaena_healthyLeaf2",
    "Lily_healthyLeaf",
    "Lily_healthyLeaf2",
    "Mint_healthyLeaf",
    "Mint_wiltLeaf",
    "Orchid_healthyFlower",
    "Orchid_healthyFlower2",
    "PeaceLily_healthyLeaf",
    "PeaceLily_yellowLeaf",
    "Pothos_healthyLeaf",
    "Pothos_yellowLeaf",
]

DISEASE_DATABASE = {
    "Aloevera_healthyLeaf": {
        "disease": "Healthy",
        "plant": "Aloe Vera",
        "symptoms": "No signs of disease or stress. Leaves are firm and green.",
        "treatment": "Continue regular care. Water every 2-3 weeks and provide bright indirect sunlight.",
    },
    "Aloevera_sunstressLeaf": {
        "disease": "Sun Stress",
        "plant": "Aloe Vera",
        "symptoms": "Leaf tips turning brown or reddish due to excessive sun exposure.",
        "treatment": "Move to a location with bright indirect light. Avoid direct midday sun.",
    },
    "Cactus_healthyCactus": {
        "disease": "Healthy",
        "plant": "Cactus",
        "symptoms": "No signs of disease. Plant looks firm and healthy.",
        "treatment": "Water sparingly every 2-4 weeks. Ensure well-draining soil.",
    },
    "Cactus_healthyCactus2": {
        "disease": "Healthy",
        "plant": "Cactus",
        "symptoms": "No signs of disease. Plant looks firm and healthy.",
        "treatment": "Water sparingly every 2-4 weeks. Ensure well-draining soil.",
    },
    "Dracaena_healthyLeaf": {
        "disease": "Healthy",
        "plant": "Dracaena",
        "symptoms": "No disease signs. Leaves are green and growing well.",
        "treatment": "Water when top inch of soil is dry. Avoid fluoride in water.",
    },
    "Dracaena_healthyLeaf2": {
        "disease": "Healthy",
        "plant": "Dracaena",
        "symptoms": "No disease signs. Leaves are green and growing well.",
        "treatment": "Water when top inch of soil is dry. Avoid fluoride in water.",
    },
    "Lily_healthyLeaf": {
        "disease": "Healthy",
        "plant": "Lily",
        "symptoms": "No signs of disease. Leaves are vibrant green.",
        "treatment": "Keep soil moist, provide bright indirect light and fertilize monthly.",
    },
    "Lily_healthyLeaf2": {
        "disease": "Healthy",
        "plant": "Lily",
        "symptoms": "No signs of disease. Leaves are vibrant green.",
        "treatment": "Keep soil moist, provide bright indirect light and fertilize monthly.",
    },
    "Mint_healthyLeaf": {
        "disease": "Healthy",
        "plant": "Mint",
        "symptoms": "No signs of disease. Leaves are fragrant and green.",
        "treatment": "Keep soil consistently moist. Prune regularly to encourage bushy growth.",
    },
    "Mint_wiltLeaf": {
        "disease": "Wilting",
        "plant": "Mint",
        "symptoms": "Leaves drooping and wilting. May be caused by underwatering or root rot.",
        "treatment": "Check soil moisture. If dry, water thoroughly. If overwatered, improve drainage and reduce watering.",
    },
    "Orchid_healthyFlower": {
        "disease": "Healthy",
        "plant": "Orchid",
        "symptoms": "No signs of disease. Flowers and leaves look vibrant.",
        "treatment": "Water once a week, use well-draining orchid mix, and provide bright indirect light.",
    },
    "Orchid_healthyFlower2": {
        "disease": "Healthy",
        "plant": "Orchid",
        "symptoms": "No signs of disease. Flowers and leaves look vibrant.",
        "treatment": "Water once a week, use well-draining orchid mix, and provide bright indirect light.",
    },
    "PeaceLily_healthyLeaf": {
        "disease": "Healthy",
        "plant": "Peace Lily",
        "symptoms": "No signs of disease. Leaves are dark green and glossy.",
        "treatment": "Keep soil moist, avoid overwatering. Tolerates low light conditions.",
    },
    "PeaceLily_yellowLeaf": {
        "disease": "Yellowing",
        "plant": "Peace Lily",
        "symptoms": "Yellow leaves indicating overwatering, nutrient deficiency, or root issues.",
        "treatment": "Reduce watering frequency. Check for root rot and repot if necessary. Add balanced fertilizer.",
    },
    "Pothos_healthyLeaf": {
        "disease": "Healthy",
        "plant": "Pothos",
        "symptoms": "No signs of disease. Leaves are green and healthy.",
        "treatment": "Water every 1-2 weeks and provide moderate indirect light.",
    },
    "Pothos_yellowLeaf": {
        "disease": "Yellowing",
        "plant": "Pothos",
        "symptoms": "Yellow leaves due to overwatering, underwatering, or lack of light.",
        "treatment": "Ensure proper drainage, adjust watering schedule, and provide adequate indirect light.",
    },
}

_model = None

def get_model():
    global _model
    if _model is None:
        import tensorflow as tf
        print(f"Loading TensorFlow model from {MODEL_PATH} ...")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    return _model

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        img: Image.Image | None = None

        if request.content_type and "multipart/form-data" in request.content_type:
            if "image" not in request.files:
                return jsonify({"error": "No image file in request"}), 400
            file = request.files["image"]
            img = Image.open(file.stream)
        else:
            data = request.get_json(force=True, silent=True) or {}
            b64 = data.get("image", "")
            if not b64:
                return jsonify({"error": "No image data provided"}), 400
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes))

        input_arr = preprocess_image(img)
        model = get_model()
        preds = model.predict(input_arr, verbose=0)[0]

        top_idx = int(np.argmax(preds))
        confidence = float(preds[top_idx]) * 100.0
        class_name = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else "Unknown"
        info = DISEASE_DATABASE.get(
            class_name,
            {
                "disease": class_name,
                "plant": "Unknown",
                "symptoms": "No information available.",
                "treatment": "Please consult a plant expert.",
            },
        )

        top5_indices = np.argsort(preds)[::-1][:5]
        top5 = [
            {
                "class": CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}",
                "confidence": round(float(preds[i]) * 100, 2),
            }
            for i in top5_indices
        ]

        return jsonify(
            {
                "class": class_name,
                "plant": info["disease"],
                "disease": info["disease"],
                "confidence": round(confidence, 2),
                "symptoms": info["symptoms"],
                "treatment": info["treatment"],
                "top5": top5,
            }
        )

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": _model is not None})

if __name__ == "__main__":
    get_model()

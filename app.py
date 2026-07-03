"""
app.py
------
OptiCrop - Smart Agricultural Production Optimization Engine
Flask backend (Epic 5: Application Building)

Routes:
    GET  /              -> Home page
    GET  /about          -> About page
    GET  /findyourcrop    -> Crop prediction form
    POST /predict         -> Handles form submission, runs model.predict(), shows result
"""

import os
import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_DIR = "model"

# ---- Load the trained model, scaler, and label encoder at startup ----
model_path = os.path.join(MODEL_DIR, "model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path)):
    raise FileNotFoundError(
        "Model files not found in model/. Run `python train_model.py` first "
        "(and `python generate_dataset.py` before that if you have no dataset)."
    )

with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
with open(encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Short human-friendly blurbs shown alongside the predicted crop
CROP_INFO = {
    "rice": "Thrives in warm, humid conditions with high rainfall and standing water.",
    "maize": "A versatile cereal crop that prefers warm days and moderate rainfall.",
    "chickpea": "A cool-season legume that tolerates dry conditions well.",
    "kidneybeans": "Grows best in cool, moderately humid conditions with well-drained soil.",
    "pigeonpeas": "A hardy, drought-tolerant legume suited to warm climates.",
    "mothbeans": "Extremely drought-resistant, ideal for arid and semi-arid regions.",
    "mungbean": "A fast-growing legume that prefers warm weather and moderate rainfall.",
    "blackgram": "Prefers warm, humid climates with well-distributed rainfall.",
    "lentil": "A cool-season crop that grows well in moderate rainfall regions.",
    "pomegranate": "Prefers a semi-arid climate with hot summers and mild winters.",
    "banana": "Needs consistently warm temperatures and high humidity year-round.",
    "mango": "Thrives in tropical and subtropical climates with a dry flowering period.",
    "grapes": "Prefers warm, dry conditions with well-drained, nutrient-rich soil.",
    "watermelon": "Loves heat, sunlight, and consistent moisture during growth.",
    "muskmelon": "Prefers warm temperatures and low humidity for best fruit quality.",
    "apple": "Needs a cold winter chill period and cool, temperate growing conditions.",
    "orange": "Grows well in subtropical climates with moderate rainfall.",
    "papaya": "Prefers hot, humid, frost-free environments year-round.",
    "coconut": "Needs a hot, humid, coastal-like climate with abundant rainfall.",
    "cotton": "Prefers warm temperatures and a long frost-free growing season.",
    "jute": "Needs high humidity and heavy rainfall during the growing season.",
    "coffee": "Grows best at higher elevations with moderate temperatures and rainfall.",
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/findyourcrop")
def find_your_crop():
    return render_template("findyourcrop.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Epic 5, Story 3: collect and convert user inputs
        values = [float(request.form.get(feat, 0)) for feat in FEATURES]
        input_array = np.array(values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction_idx = model.predict(input_scaled)[0]
        crop_name = label_encoder.inverse_transform([prediction_idx])[0]

        # Confidence score, if the underlying model supports predict_proba
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_scaled)[0]
            confidence = round(float(np.max(proba)) * 100, 2)

        info = CROP_INFO.get(crop_name, "A crop well suited to the conditions you provided.")

        return render_template(
            "findyourcrop.html",
            prediction=crop_name.capitalize(),
            confidence=confidence,
            info=info,
            form_values=dict(zip(FEATURES, values)),
        )
    except Exception as exc:  # noqa: BLE001
        return render_template(
            "findyourcrop.html",
            error=f"Could not generate a prediction: {exc}",
        )


if __name__ == "__main__":
    # Epic 5: run the Flask development server
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
from PIL import Image
import os

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

app = Flask(__name__)
app.secret_key = "pollen-prof-secret"

IMG_SIZE = (128, 128)
MODEL_PATH = os.path.join("saved_model", "pollen_model.h5")

CLASS_NAMES = [f"Pollen_{i}" for i in range(23)]

class DummyModel:
    def predict(self, arr):
        probs = np.random.rand(1, len(CLASS_NAMES))
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

def load_model():
    if TF_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print("Failed to load model, falling back to DummyModel:", e)
    else:
        if not TF_AVAILABLE:
            print("TensorFlow not available — using DummyModel.")
        else:
            print("Model file not found — using DummyModel.")
    return DummyModel()

model = load_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        flash("Please upload an image.")
        return redirect(url_for("index"))

    file = request.files["file"]
    try:
        img = Image.open(file).convert("RGB").resize(IMG_SIZE)
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)
        label = CLASS_NAMES[int(np.argmax(preds))]
        prob = float(np.max(preds))
        return render_template("result.html", label=label, prob=f"{prob:.2%}")
    except Exception as e:
        flash(f"Failed to process image: {e}")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Recreate the exact model architecture from training
def create_model():
    from tensorflow.keras.layers import Input, Rescaling, RandomRotation, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.models import Sequential
    
    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(256, 256, 3)
    )
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False
    
    model = Sequential([
        Input(shape=(256, 256, 3)),
        Rescaling(1./255),
        RandomRotation(0.1),
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    return model

try:
    model = create_model()
    model.load_weights("best_densenet_feature_extraction (3).keras")
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = create_model()
    print("Using untrained model...")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((256, 256))
        img = np.array(image)  # Don't normalize - model has Rescaling layer
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)[0][0]
        result = "Pneumonia" if prediction > 0.5 else "Normal"
        return jsonify({"prediction": result, "confidence": round(float(prediction), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

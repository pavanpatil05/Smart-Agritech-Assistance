from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "plant_disease_mobilenetv2.h5"
CLASS_PATH = "class_names.json"

# ✅ Load model once
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

PLANT_CLASS_MAP = {
    "grape": [i for i, c in enumerate(class_names) if "Grape" in c],
    "onion": [i for i, c in enumerate(class_names) if "Onion" in c],
    "tomato": [i for i, c in enumerate(class_names) if "Tomato" in c],
}

IMG_SIZE = (224, 224)


def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.post("/predict/")
async def predict(file: UploadFile = File(...), plant_type: str = Form(...)):

    contents = await file.read()

    # ✅ FIX 1: Force fresh image load every time
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    processed_image = preprocess_image(image)

    # ✅ FIX 2: Thread-safe prediction
    predictions = model(processed_image, training=False).numpy()[0]

    plant_type = plant_type.lower()

    if plant_type not in PLANT_CLASS_MAP:
        return {"error": "Invalid plant type"}

    valid_indices = PLANT_CLASS_MAP[plant_type]

    overall_best_index = np.argmax(predictions)

    if overall_best_index not in valid_indices:
        return {
            "success": False,
            "error": f"Upload valid {plant_type} leaf image"
        }

    filtered_preds = {i: float(predictions[i]) for i in valid_indices}
    best_index = max(filtered_preds, key=filtered_preds.get)

    predicted_class = class_names[best_index]
    confidence = filtered_preds[best_index] * 100

    return {
        "success": True,
        "plant": plant_type,
        "prediction": predicted_class,
        "confidence": round(confidence, 2)
    }
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Form
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os



MODEL_PATH = "plant_disease_mobilenetv2.h5"
CLASS_PATH = "class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

# ✅ NOW create mapping (after loading classes)
PLANT_CLASS_MAP = {
    "grape": [i for i, c in enumerate(class_names) if "Grape" in c],
    "onion": [i for i, c in enumerate(class_names) if "Onion" in c],
    "tomato": [i for i, c in enumerate(class_names) if "Tomato" in c],
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change to frontend domain in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_PATH = "plant_disease_mobilenetv2.h5"
CLASS_PATH = "class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

IMG_SIZE = (224, 224)


@app.get("/")
def home():
    return {"status": "Plant Disease API Running 🚀"}


def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array




@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    plant_type: str = Form(...)
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)[0]

    # Get indices for selected plant
    plant_type = plant_type.lower()

    if plant_type not in PLANT_CLASS_MAP:
        return {"error": "Invalid plant type. Choose from grape/onion/tomato"}

    valid_indices = PLANT_CLASS_MAP[plant_type]

    # Filter predictions
    filtered_preds = {i: predictions[i] for i in valid_indices}

    # Get best prediction from filtered
    best_index = max(filtered_preds, key=filtered_preds.get)

    predicted_class = class_names[best_index]
    confidence = float(predictions[best_index]) * 100

    return {
        "plant": plant_type,
        "prediction": predicted_class,
        "confidence": f"{confidence:.2f}%"
    }
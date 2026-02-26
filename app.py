from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change to frontend domain in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 2️⃣ Load Model
# -----------------------------
MODEL_PATH = "plant_disease_mobilenetv2.h5"
CLASS_PATH = "class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

IMG_SIZE = (224, 224)

# -----------------------------
# 3️⃣ Health Check Route
# -----------------------------
@app.get("/")
def home():
    return {"status": "Plant Disease API Running 🚀"}

# -----------------------------
# 4️⃣ Image Preprocessing
# -----------------------------
def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# 5️⃣ Prediction Route
# -----------------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)[0]

    best_index = np.argmax(predictions)
    predicted_class = class_names[best_index]
    confidence = float(predictions[best_index]) * 100

    return {
        "prediction": predicted_class,
        "confidence": f"{confidence:.2f}%"
    }
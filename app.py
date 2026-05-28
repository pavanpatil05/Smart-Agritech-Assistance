from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import json
from fastapi import FastAPI

app = FastAPI()
print("App started")

@app.get("/")

def home():

    return {"message": "API is running"}

MODEL_PATH = "model2.h5"

try:

    print("Loading model...")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print("Model loaded successfully")

except Exception as e:

    print("MODEL ERROR:", e)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model2.h5"
CLASS_PATH = "class_names.json"

# ✅ Load model once
# model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(CLASS_PATH, "r") as f:

    class_names = json.load(f)

# Marathi Disease Names

with open("marathi_diseases.json", "r", encoding="utf-8") as f:
    MARATHI_DISEASES = json.load(f)

with open("marathi_solutions.json", "r", encoding="utf-8") as f:
    MARATHI_SOLUTIONS = json.load(f)

PLANT_CLASS_MAP = {
    "grape": [i for i, c in enumerate(class_names) if "Grape" in c],
    "onion": [i for i, c in enumerate(class_names) if "Onion" in c],
    "tomato": [i for i, c in enumerate(class_names) if "Tomato" in c],
}

IMG_SIZE = (224, 224)


def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img)
    
    img = preprocess_input(img)   # 🔥 MUST MATCH TRAINING
    img = np.expand_dims(img, axis=0)
    return img




# @app.post("/predict/")
# async def predict(file: UploadFile = File(...), plant_type: str = Form(...)):

#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents)).convert("RGB")

#     processed_image = preprocess_image(image)

#     predictions = model(processed_image, training=False).numpy()[0]

#     plant_type = plant_type.lower()

#     if plant_type not in PLANT_CLASS_MAP:
#         return {"success": False, "error": "Invalid plant type"}

#     valid_indices = PLANT_CLASS_MAP[plant_type]

#     filtered_preds = {i: float(predictions[i]) for i in valid_indices}
#     best_index = max(filtered_preds, key=filtered_preds.get)

#     confidence = filtered_preds[best_index]

#     #✅ Confidence check
#     if confidence < 0.3:
#         return {
#             "success": False,
#             "error": "Low confidence. Try better image"
#         }

#     predicted_class = class_names[best_index]

#     marathi_disease = MARATHI_DISEASES.get(
#     predicted_class,
#     predicted_class
#     )
#     marathi_solution = MARATHI_SOLUTIONS.get(
#     predicted_class,
#     "योग्य उपाय उपलब्ध नाही"
#     )
 
#     return {
#     "success": True,
#     "वनस्पती": plant_type,
#     # "disease_english": predicted_class,
#     "रोग": marathi_disease,
#     "उपाय": marathi_solution,
#     "confidence": round(confidence * 100, 2)
#     }



@app.post("/predict/")
async def predict(file: UploadFile = File(...), plant_type: str = Form(...)):
    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        processed_image = preprocess_image(image)

        predictions = model(processed_image, training=False).numpy()[0]

        plant_type = plant_type.lower()

        if plant_type not in PLANT_CLASS_MAP:
            return {"success": False, "error": "Invalid plant type"}

        valid_indices = PLANT_CLASS_MAP[plant_type]

        filtered_preds = {i: float(predictions[i]) for i in valid_indices}

        best_index = max(filtered_preds, key=filtered_preds.get)

        confidence = filtered_preds[best_index]

        if confidence < 0.3:
            return {
                "success": False,
                "error": "Low confidence. Try better image"
            }

        predicted_class = class_names[best_index]

        marathi_disease = MARATHI_DISEASES.get(
            predicted_class,
            predicted_class
        )

        marathi_solution = MARATHI_SOLUTIONS.get(
            predicted_class,
            "योग्य उपाय उपलब्ध नाही"
        )

        return {
            "success": True,
            "वनस्पती": plant_type,
            "रोग": marathi_disease,
            "उपाय": marathi_solution,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        print("PREDICTION ERROR:", str(e))
        return {
            "success": False,
            "error": str(e)
        }
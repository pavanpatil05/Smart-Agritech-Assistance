from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import json
import os
import keras
from tensorflow.keras.models import load_model
app = FastAPI()
print("App started")

MODEL_PATH = "best_model.keras"
model = None

@app.on_event("startup")
async def load_my_model():
    global model

    import os
    import traceback
    print("========== STARTUP ==========")
    print("Current directory:", os.getcwd())
    print("Files:", os.listdir("."))

    try:
        print("Loading best_model.keras ...")
        model = keras.models.load_model(
            "best_model.keras",
            compile=False
        )
        print("✅ MODEL LOADED SUCCESSFULLY")
    except Exception as e:
        print("❌ MODEL LOAD FAILED")
        print("ERROR:", str(e))
        traceback.print_exc()

@app.get("/")
def home():
    return {"message": "API is running"}



@app.get("/load-model")
def load_model_debug():
    global model

    try:
        model = keras.models.load_model(
            "best_model.keras",
            compile=False
        )
        return {
            "success": True
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/test-model")
def test_model():

    if model is None:
        return {
            "status": "failed"
        }

    return {
        "status": "loaded"
    }
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_PATH = "class_names.json"

# ✅ Load model once
# model = tf.keras.models.load_model(MODEL_PATH, compile=False)
# Load class names safely
try:
    with open(CLASS_PATH, "r") as f:
        class_names = json.load(f)

    print("✅ class_names loaded")

except Exception as e:
    print("❌ CLASS FILE ERROR:", str(e))
    class_names = []

# Marathi disease names
try:
    with open("marathi_diseases.json", "r", encoding="utf-8") as f:
        MARATHI_DISEASES = json.load(f)

    print("✅ Marathi diseases loaded")

except Exception as e:
    print("❌ Marathi disease file error:", str(e))
    MARATHI_DISEASES = {}

# Marathi solutions
try:
    with open("marathi_solutions.json", "r", encoding="utf-8") as f:
        MARATHI_SOLUTIONS = json.load(f)

    print("✅ Marathi solutions loaded")

except Exception as e:
    print("❌ Marathi solution file error:", str(e))
    MARATHI_SOLUTIONS = {}

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




@app.post("/predict/")
async def predict(file: UploadFile = File(...), plant_type: str = Form(...)):
    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        processed_image = preprocess_image(image)

        if model is None:
            return{
                "success": False,
                "error": "Model not loaded"
            }

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


@app.get("/debug")

def debug():

    import keras

    return {

        "keras_file": str(keras.__file__),

        "keras_version": keras.__version__,

        "has_models": hasattr(keras, "models"),

        "has_model": hasattr(keras, "model")

    }

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


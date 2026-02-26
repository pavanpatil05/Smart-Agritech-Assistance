from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load model
model = tf.keras.models.load_model("plant_disease_mobilenetv2.h5")

IMG_SIZE = (224, 224)

class_names = ['Grape_Anthracnose leaf', 'Grape_Brown spot leaf', 'Grape_Downy mildew leaf', 
'Grape_Healthy_leaf', 'Grape_Mites_leaf disease', 'Grape_Powdery mildew leaf',
'Grape_shot hole leaf disease', 'Onion_Alternaria_D', 'Onion_Botrytis Leaf Blight', 
'Onion_Bulb Rot', 'Onion_Bulb_blight-D', 'Onion_Caterpillar-P', 'Onion_Downy mildew',
'Onion_Fusarium-D', 'Onion_Healthy leaves', 'Onion_Iris yellow virus_augment', 
'Onion_Purple blotch', 'Onion_Rust', 'Onion_Virosis-D', 'Onion_Xanthomonas Leaf Blight', 
'Onion_stemphylium Leaf Blight', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']  


def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100

    return {
        "prediction": predicted_class,
        "confidence": f"{confidence:.2f}%"
    }
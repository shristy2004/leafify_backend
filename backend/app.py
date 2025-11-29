from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from datetime import datetime
import numpy as np

from db import predictions
from utils.preprocess import preprocess_image

app = FastAPI()

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = "model/leaf_model.h5"
model = load_model(MODEL_PATH)

CLASS_NAMES = ["Healthy", "Early Blight", "Late Blight", "Leaf Mold"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image bytes
    image_bytes = await file.read()

    # Preprocess image
    img = preprocess_image(image_bytes)     # Must return shape (1, 224, 224, 3)

    # Model prediction
    pred = model.predict(img)[0]
    confidence = float(np.max(pred))
    label = CLASS_NAMES[int(np.argmax(pred))]

    # Save to MongoDB
    doc = {
        "file_name": file.filename,
        "prediction": label,
        "confidence": confidence,
        "timestamp": datetime.utcnow(),
    }

    await predictions.insert_one(doc)

    return {
        "label": label,
        "confidence": confidence,
        "file_name": file.filename
    }

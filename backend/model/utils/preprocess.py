import cv2
import numpy as np

def preprocess_image(image_bytes):
    """
    Convert raw image bytes → model-ready input.
    """

    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode image (OpenCV reads it as BGR)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize (your model expects 224x224 — change if needed)
    img = cv2.resize(img, (224, 224))

    # Convert BGR → RGB (TensorFlow uses RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to range 0–1
    img = img / 255.0

    # Expand dimensions → (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)

    return img

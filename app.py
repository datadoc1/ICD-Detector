# app.py

import gradio as gr
import torch
from PIL import Image

# --- 1. LOAD THE MODEL ---
# Load the model from the .pt file you got from Linh.
# trust_repo=True is important for custom models.
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', trust_repo=True)

# --- 2. DEFINE THE PREDICTION FUNCTION ---
# This function takes an image as input and returns the image with boxes drawn on it.
def predict_image(image):
    # Convert the input image to a PIL Image (if it's not already)
    img = Image.fromarray(image.astype('uint8'), 'RGB')
    
    # Run inference
    results = model(img)
    
    # The 'render()' method draws the bounding boxes on the image
    # and returns a list of annotated images. We'll take the first one.
    annotated_image = results.render()[0]
    
    return annotated_image

# --- 3. CREATE THE GRADIO INTERFACE ---
# This creates the simple "drag-and-drop" web UI.
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Upload Chest X-Ray"),
    outputs=gr.Image(type="numpy", label="Model Prediction"),
    title="ICD Device Detector",
    description="Upload an X-ray to identify the ICD model. This is a demo based on Linh Nguyen's research with Dr. Clark."
)

# --- 4. LAUNCH THE APP ---
# This line starts the web server.
iface.launch()
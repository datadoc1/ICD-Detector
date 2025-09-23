# app.py

import gradio as gr
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

# --- 1. LOAD THE MODEL ---
model = YOLO('best.pt')

# --- 2. DEFINE THE PREDICTION FUNCTION ---
def predict_image(image):
    # Convert input to RGB if needed
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    img = Image.fromarray(image.astype('uint8'), 'RGB')

    # Run inference
    results = model(img)

    # Draw boxes on the image
    annotated_image = np.array(img)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_map = {
                0: 'BSC120',
                1: 'BSC140',
                2: 'Biotronik',
                3: 'Boston Scientific',
                4: 'Medtronic',
                5: 'St Jude'
            }
            label = f"{class_map.get(cls, f'Unknown({cls})')} {conf:.2f}"
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(annotated_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Collect mapped class names
    mapped_names = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0].item())
            mapped_names.append(class_map.get(cls, f"Unknown({cls})"))
    mapped_names_str = ", ".join(set(mapped_names)) if mapped_names else "No device detected"

    return annotated_image, mapped_names_str

# --- 3. CREATE THE GRADIO INTERFACE ---
# This creates the simple "drag-and-drop" web UI.
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Upload Chest X-Ray"),
    outputs=[
        gr.Image(type="numpy", label="Model Prediction"),
        gr.Textbox(label="Predicted Device(s)")
    ],
    title="ICD Device Detector",
    description="Upload an X-ray to identify the ICD model. This is a demo based on Linh Nguyen's research with Dr. Clark."
)

# --- 4. LAUNCH THE APP ---
# This line starts the web server.
iface.launch()
# app.py

import gradio as gr
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os
import random

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

# --- 2b. RANDOM IMAGE HANDLER ---
def random_cxr():
    images_dir = "images"
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        return None, "No images found in /images"
    random_file = random.choice(image_files)
    img_path = os.path.join(images_dir, random_file)
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    return predict_image(img_np)

# --- 3. CREATE THE GRADIO INTERFACE ---
# This creates the "drag-and-drop" web UI with a Random CXR button.

with gr.Blocks() as iface:
    gr.Markdown("# ICD Device Detector")
    gr.Markdown(
        """
        **Upload an X-ray or click "Random CXR" to identify the ICD model.**
        
        This demo uses a deep learning model trained on chest X-rays to detect and classify implantable cardioverter-defibrillator (ICD) devices.
        The model is based on YOLOv8 and was developed as part of Linh Nguyen's research with Dr. Clark.
        
        **Average inference time is ~30 seconds due to heavy server usage. Please be patient after submitting an image.**
        
        This tool is intended for research and educational purposes only.
        """
    )

    with gr.Row():
        img_input = gr.Image(type="numpy", label="Upload Chest X-Ray")
        random_btn = gr.Button("Random CXR")

    with gr.Row():
        base_viewer = gr.Image(type="numpy", label="Base Image", interactive=False)
        output_img = gr.Image(type="numpy", label="Detection", interactive=False)

    pred_bar = gr.Markdown("")

    def predict_image_with_bar(image):
        if image is None:
            return None, None, ""
        pred_img, pred_txt, pred_bar_str = _predict_and_bar(image)
        return image, pred_img, pred_bar_str

    def random_cxr_with_bar():
        images_dir = "images"
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            return None, None, "No images found in /images"
        random_file = random.choice(image_files)
        img_path = os.path.join(images_dir, random_file)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        pred_img, pred_txt, pred_bar_str = _predict_and_bar(img_np)
        return img_np, pred_img, pred_bar_str

    def _predict_and_bar(image):
        # Use original predict_image logic, but also extract top prediction and confidence
        annotated_image, mapped_names_str = predict_image(image)
        # Extract top prediction and confidence
        results = model(Image.fromarray(image.astype('uint8'), 'RGB'))
        top_pred = None
        top_conf = None
        class_map = {
            0: 'BSC120',
            1: 'BSC140',
            2: 'Biotronik',
            3: 'Boston Scientific',
            4: 'Medtronic',
            5: 'St Jude'
        }
        for r in results:
            if len(r.boxes) > 0:
                # Take the highest confidence box
                confs = [float(box.conf[0].item()) for box in r.boxes]
                idx = int(np.argmax(confs))
                box = r.boxes[idx]
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                top_pred = class_map.get(cls, f"Unknown({cls})")
                top_conf = int(conf * 100)
                break
        if top_pred is not None and top_conf is not None:
            bar_str = f"<div style='text-align:center; font-size:1.2em; font-weight:bold;'>This is a {top_pred} ICD, and we are {top_conf}% confident about it</div>"
        else:
            bar_str = "<div style='text-align:center; font-size:1.2em; font-weight:bold;'>No ICD detected</div>"
        return annotated_image, mapped_names_str, bar_str

    img_input.change(
        fn=predict_image_with_bar,
        inputs=img_input,
        outputs=[base_viewer, output_img, pred_bar]
    )
    random_btn.click(
        fn=random_cxr_with_bar,
        inputs=None,
        outputs=[base_viewer, output_img, pred_bar]
    )

# --- 4. LAUNCH THE APP ---
# This line starts the web server.
iface.launch()
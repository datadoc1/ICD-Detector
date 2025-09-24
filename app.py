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

# --- 3. CREATE THE GRADIO INTERFACE ---
# This creates the "drag-and-drop" web UI with a Random CXR button.

with gr.Blocks() as iface:
    gr.Markdown("What Kind of ICD is This?")
    gr.Markdown(
        """
        **Upload an X-ray or click <span style='font-size:1.2em;'>🎲</span> Random CXR to identify the ICD model.**
        
        This demo uses a deep learning model trained on chest X-rays to detect and classify implantable cardioverter-defibrillator (ICD) devices.
        The model is based on YOLOv8 and was developed as part of Linh Nguyen's research with Dr. Kal Clark.
        
        This tool is intended for research and educational purposes only.
        """
    )

    with gr.Row():
        img_input = gr.Image(type="numpy", label="Upload Chest X-Ray", show_label=False, show_download_button=False, elem_id="upload-img")
        random_btn = gr.Button("🎲 Random CXR", elem_id="random-btn", variant="primary")

    with gr.Row():
        base_viewer = gr.Image(type="numpy", label="Base Image", interactive=False, elem_id="base-img")
        output_img = gr.Image(type="numpy", label="Detection", interactive=False, elem_id="detect-img")

    pred_bar = gr.Markdown("")

    # Style buttons for consistency
    gr.HTML(
        '''
        <style>
        #upload-img .upload-box { border-radius: 8px; }
        #random-btn button { background: #007bff; color: white; border-radius: 8px; font-weight: bold; }
        .gr-button { border-radius: 8px !important; font-weight: bold !important; }
        </style>
        '''
    )

    # Show base image immediately on upload
    def show_base_image(image):
        return image

    img_input.change(
        fn=show_base_image,
        inputs=img_input,
        outputs=base_viewer,
        queue=False
    )

    # Run prediction after base image is shown
    def predict_image_with_bar(image):
        if image is None:
            return None, ""
        pred_img, pred_txt, pred_bar_str = _predict_and_bar(image)
        return pred_img, pred_bar_str

    img_input.change(
        fn=predict_image_with_bar,
        inputs=img_input,
        outputs=[output_img, pred_bar],
        queue=True
    )

    # Random CXR: show base image first, then run prediction
    def random_cxr_base():
        images_dir = "images"
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            return None
        random_file = random.choice(image_files)
        img_path = os.path.join(images_dir, random_file)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        return img_np

    random_btn.click(
        fn=random_cxr_base,
        inputs=None,
        outputs=base_viewer,
        queue=False
    )

    def random_cxr_predict():
        images_dir = "images"
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            return None, None
        random_file = random.choice(image_files)
        img_path = os.path.join(images_dir, random_file)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        pred_img, pred_txt, pred_bar_str = _predict_and_bar(img_np)
        return pred_img, pred_bar_str

    random_btn.click(
        fn=random_cxr_predict,
        inputs=None,
        outputs=[output_img, pred_bar],
        queue=True
    )

    def _predict_and_bar(image):
        # Run model only once and reuse results for both annotation and confidence
        img = Image.fromarray(image.astype('uint8'), 'RGB')
        results = model(img)
        # Draw boxes and collect mapped class names
        annotated_image = np.array(img)
        class_map = {
            0: 'BSC120',
            1: 'BSC140',
            2: 'Biotronik',
            3: 'Boston Scientific',
            4: 'Medtronic',
            5: 'St Jude'
        }
        mapped_names = []
        top_pred = None
        top_conf = None
        for r in results:
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                mapped_names.append(class_map.get(cls, f"Unknown({cls})"))
                label = f"{class_map.get(cls, f'Unknown({cls})')} {conf:.2f}"
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(annotated_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            if len(r.boxes) > 0:
                # Take the highest confidence box for bar
                confs = [float(box.conf[0].item()) for box in r.boxes]
                idx = int(np.argmax(confs))
                box = r.boxes[idx]
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                top_pred = class_map.get(cls, f"Unknown({cls})")
                top_conf = int(conf * 100)
        mapped_names_str = ", ".join(set(mapped_names)) if mapped_names else "No device detected"
        if top_pred is not None and top_conf is not None:
            bar_str = f"<div style='text-align:center; font-size:1.2em; font-weight:bold;'>This is a {top_pred} ICD, and we are {top_conf}% confident about it</div>"
        else:
            bar_str = "<div style='text-align:center; font-size:1.2em; font-weight:bold;'>No ICD detected</div>"
        return annotated_image, mapped_names_str, bar_str


# --- 4. LAUNCH THE APP ---
# This line starts the web server.
iface.launch()
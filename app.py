# app.py

import gradio as gr
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os
import random
import torch

# Optimize for CPU multi-threading
torch.set_num_threads(os.cpu_count())

# Check if CUDA is available and set device
device = 'cpu'  # Force CPU as per requirements
print(f"Using device: {device}")

# --- 1. LOAD THE MODEL ---
model = YOLO('best.pt')

# Load model explicitly on CPU
model.to(device)


# Cache image files to avoid repeated directory scans
images_dir = "images"
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# --- 2. DEFINE THE PREDICTION FUNCTION ---

# --- 3. CREATE THE GRADIO INTERFACE ---
# This creates the "drag-and-drop" web UI with a Random CXR button.

with gr.Blocks() as iface:
    gr.Markdown("What model device is on this CXR?:")
    gr.Markdown(
        """
        **Upload an X-ray or click <span style='font-size:1.2em;'>🎲</span> Random CXR to identify the device model.**
        
        This demo uses a deep learning model trained on chest X-rays to detect and classify implantable cardiac devices.
        The model is based on YOLOv8 and was developed as part of Linh Nguyen's research with Dr. Kal Clark.
        
        This tool is intended for research and educational purposes only.
        """
    )

    # Supported models table
    supported_models = [
        'Biotronik',
        'Biotronik - Birdpeak can',
        'BSC120 MRI-nonconditional ICD',
        'BSC140 MRI-conditional ICD',
        'Boston Scientific',
        'Boston Scientific BOS112',
        'Boston Scientific BSC120',
        'Boston Scientific BSC140',
        'Medtronic',
        'Medtronic - PSI',
        'Medtronic - PUG',
        'Medtronic - PVR',
        'Medtronic - PXR',
        'Medtronic - PXT',
        'St Jude',
        'St Jude - Current DR',
        'St Jude - Unify'
    ]
    table_md = "| Supported Models |\n| --- |\n"
    for model in supported_models:
        table_md += f"| {model} |\n"
    gr.Markdown(table_md)

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

    # Function to select a random image from the cached list
    def get_random_image():
        if not image_files:
            return None
        random_file = random.choice(image_files)
        img_path = os.path.join(images_dir, random_file)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        return img_np

    # Show base image and run prediction in one go to avoid redundant processing
    def process_image(image):
        if image is None:
            return None, None, ""
        # Run prediction only once
        pred_img, pred_txt, pred_bar_str = _predict_and_bar(image)
        return image, pred_img, pred_bar_str

    img_input.change(
        fn=process_image,
        inputs=img_input,
        outputs=[base_viewer, output_img, pred_bar],
        queue=True
    )

    # Random CXR: show base image and run prediction in one function
    def random_cxr_process():
        img_np = get_random_image()
        if img_np is None:
            return None, None, ""
        # Run prediction only once
        pred_img, pred_txt, pred_bar_str = _predict_and_bar(img_np)
        return img_np, pred_img, pred_bar_str

    random_btn.click(
        fn=random_cxr_process,
        inputs=None,
        outputs=[base_viewer, output_img, pred_bar],
        queue=True
    )

    def _predict_and_bar(image):
        # Run model only once and reuse results for both annotation and confidence
        import time
        start_time = time.time()
        # Use numpy directly for model input (faster than PIL conversion)
        results = model(image, device=device)
        inference_end = time.time()
        inference_time = inference_end - start_time
        print(f"Inference time: {inference_time:.2f}s")
        # Draw boxes and collect mapped class names
        annotated_image = image.copy()  # Use copy to avoid modifying input
        class_map = {
            0: 'Biotronik',
            1: 'Biotronik - Birdpeak can',
            2: 'BSC120 MRI-nonconditional ICD',
            3: 'BSC140 MRI-conditional ICD',
            4: 'Boston Scientific',
            5: 'Boston Scientific BOS112',
            6: 'Boston Scientific BSC120',
            7: 'Boston Scientific BSC140',
            8: 'Medtronic',
            9: 'Medtronic - PSI',
            10: 'Medtronic - PUG',
            11: 'Medtronic - PVR',
            12: 'Medtronic - PXR',
            13: 'Medtronic - PXT',
            14: 'St Jude',
            15: 'St Jude - Current DR',
            16: 'St Jude - Unify'
        }
        mapped_names = []
        top_pred = None
        top_conf = None
        draw_start = time.time()
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
        draw_end = time.time()
        draw_time = draw_end - draw_start
        total_time = draw_end - start_time
        print(f"Drawing time: {draw_time:.2f}s")
        print(f"Total processing time: {total_time:.2f}s")
        mapped_names_str = ", ".join(set(mapped_names)) if mapped_names else "No device detected"
        if top_pred is not None and top_conf is not None:
            bar_str = f"<div style='text-align:center; font-size:1.2em; font-weight:bold;'>Detected device: {top_pred} with {top_conf}% confidence</div>"
        else:
            bar_str = "<div style='text-align:center; font-size:1.2em; font-weight:bold;'>The heart devices are not one of the ICDs above</div>"
        return annotated_image, mapped_names_str, bar_str


# --- 4. LAUNCH THE APP ---
# This line starts the web server.
iface.launch()
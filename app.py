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
# Prefer a cleaned project layout: model artifact is expected at model/best.pt
# Fall back to root-best.pt if the reorganized path isn't present yet.
model_path = 'model/best.pt' if os.path.exists('model/best.pt') else 'best.pt'
# Note: model was trained with a YOLOv11 architecture (training script in model/training_script.py)
model = YOLO(model_path)

# Load model explicitly on CPU (device variable set earlier); keep explicit placement
model.to(device)

# Cache image files to avoid repeated directory scans
# Prefer the cleaned dataset layout under model/data/val (common export layout).
# Fall back to the legacy top-level "images" directory if the new layout isn't present.
images_dir_candidates = ["model/data/val/images", "model/data/val", "images"]
images_dir = next((p for p in images_dir_candidates if os.path.isdir(p)), "images")
if not os.path.isdir(images_dir):
    # If nothing exists, use current directory as a last resort (no images will be found)
    images_dir = "."
image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# --- 2. DEFINE THE PREDICTION FUNCTION ---

# --- 3. CREATE THE GRADIO INTERFACE ---
# This creates the "drag-and-drop" web UI with a Random CXR button.

with gr.Blocks(css=".gr-block { max-width: 1200px; margin: 0 auto; } .gr-row { flex-wrap: wrap; }") as iface:
    gr.Markdown("# ICD Detector: Spot Implanted Cardiac Devices in Chest X-Rays.")
    gr.Markdown(
        """
        **Upload an X-ray or click <span style='font-size:1.2em;'>🎲</span> Random CXR to identify the device model.**

        **About this demo (short):**
        This project was led by Linh Nguyen — Linh performed the dataset labeling
        and the original model training (Roboflow export). Keola Ching assisted by
        adapting and hosting the model for a live demo and creating the web UI.
        Dr. Kal L. Clark served as project advisor.

        The model is a YOLOv11-based prototype trained on a small CXR dataset to
        explore automated identification of implantable cardioverter‑defibrillators (ICDs).
        Upload a CXR screenshot and the demo will attempt to detect device hardware,
        show bounding boxes, and display a confidence bar with a short interpretation.

        **Data source:** Roboflow dataset: https://universe.roboflow.com/nighthawklvn/icd-xray-image-recognition

        **Disclaimer (please read):**
        **This is for educational / entertainment purposes only — not a medical tool.**
        The prototype may confuse ICDs with pacemakers because both have similar
        radiographic appearances. Always consult official sources for MRI safety
        and clinical decisions.

        **Quick tips:**
        - Look for thick shock coils on leads to confirm an ICD.
        - If the model is uncertain, treat the result as exploratory rather than diagnostic.

        """
    )

    # Supported models accordion (collapsible to save space)
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
    models_list = "\n".join([f"- {model}" for model in supported_models])
    with gr.Accordion("Supported Models", open=False):
        gr.Markdown(f"**Supported Models:**\n{models_list}")

    with gr.Row():
        img_input = gr.Image(type="numpy", label="Upload Chest X-Ray", show_label=False, show_download_button=False, elem_id="upload-img")
        random_btn = gr.Button("🎲 Random CXR", elem_id="random-btn", variant="primary")

    with gr.Row():
        base_viewer = gr.Image(type="numpy", label="Base Image", interactive=False, elem_id="base-img")
        output_img = gr.Image(type="numpy", label="Detection", interactive=False, elem_id="detect-img")

    pred_bar = gr.Markdown("", elem_id="pred-bar")
    explain_box = gr.Markdown("", elem_id="explain-box")

    # Style buttons and result bar for consistency
    gr.HTML(
        '''
        <style>
        #upload-img .upload-box { border-radius: 8px; }
        #random-btn button { background: #007bff; color: white; border-radius: 8px; font-weight: bold; }
        .gr-button { border-radius: 8px !important; font-weight: bold !important; }

        /* Confidence bar styling */
        .icd-confidence-wrapper { width: 80%; margin: 8px auto 12px auto; background: #eee; border-radius: 10px; height: 18px; overflow: hidden; }
        .icd-confidence-fill { height: 100%; border-radius: 10px; }

        /* Result container */
        #pred-bar { text-align: center; font-size: 1.05em; font-weight: 600; margin-top: 6px; }
        #explain-box { font-size: 0.95em; color: #222; margin-top: 6px; }
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
            return None, None, "", ""
        # Run prediction only once
        pred_img, pred_txt, pred_bar_str, explain_str = _predict_and_bar(image)
        return image, pred_img, pred_bar_str, explain_str

    img_input.change(
        fn=process_image,
        inputs=img_input,
        outputs=[base_viewer, output_img, pred_bar, explain_box],
        queue=True
    )

    # Random CXR: show base image and run prediction in one function
    def random_cxr_process():
        img_np = get_random_image()
        if img_np is None:
            return None, None, "", ""
        # Run prediction only once
        pred_img, pred_txt, pred_bar_str, explain_str = _predict_and_bar(img_np)
        return img_np, pred_img, pred_bar_str, explain_str

    random_btn.click(
        fn=random_cxr_process,
        inputs=None,
        outputs=[base_viewer, output_img, pred_bar, explain_box],
        queue=True
    )

    def _predict_and_bar(image):
        # Run model only once and reuse results for both annotation and confidence + explanation
        import time
        import traceback
        start_time = time.time()
        try:
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

            # Build confidence bar and explanation
            if top_pred is not None and top_conf is not None:
                # Choose color
                if top_conf >= 80:
                    color = "#28a745"  # green
                    confidence_note = "High confidence"
                elif top_conf >= 50:
                    color = "#ffb300"  # yellow/orange
                    confidence_note = "Moderate confidence"
                else:
                    color = "#dc3545"  # red
                    confidence_note = "Low confidence"

                bar_html = f"<div style='text-align:center; font-size:1.12em; font-weight:700;'>Detected: {top_pred} — {top_conf}%</div>"
                bar_html += f"<div class='icd-confidence-wrapper'><div class='icd-confidence-fill' style='width:{top_conf}%; background:{color};'></div></div>"
                
                # Explanatory text
                if top_conf >= 80:
                    explain_html = (
                        f"**Interpretation:** Detected **{top_pred}** with high confidence ({top_conf}%). "
                        "Likely an ICD. Fun tip: look for thick shock coils on the leads to confirm an ICD visually. "
                        "Model trained on 98 images; prototype may still confuse ICDs and pacemakers."
                    )
                elif top_conf >= 50:
                    explain_html = (
                        f"**Interpretation:** Possible **{top_pred}** ({top_conf}%). "
                        "This might be a pacemaker in some cases — check for uniform leads without shock coils. "
                        "Treat this as exploratory output, not diagnostic."
                    )
                else:
                    explain_html = (
                        "**Interpretation:** Low confidence — no clear ICD detected. "
                        "Could be a pacemaker or no implanted device. Try a clearer/closer CXR or review with a specialist."
                    )

                # Add learn-more and disclaimer
                explain_html += (
                    "\n\n**Learn more & sources:** See the project README for dataset and training notes. "
                    "MRI safety reference: https://www.mrisafety.com (always consult official sources). "
                    "This demo is for education/entertainment only — not a medical device."
                )

                return annotated_image, mapped_names_str, bar_html, explain_html
            else:
                bar_str = "<div style='text-align:center; font-size:1.1em; font-weight:bold;'>No ICD from supported models detected</div>"
                explain_html = (
                    "No supported ICD models were detected with sufficient confidence. "
                    "Possible reasons: the image crop, device is a pacemaker, or no device present. "
                    "See README for device table and model limitations. This is a prototype."
                )
                return annotated_image, mapped_names_str, bar_str, explain_html

        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            annotated_image = image.copy()
            cv2.putText(annotated_image, "Prediction Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            bar_str = f"<div style='text-align:center; font-size:1.2em; font-weight:bold; color:red;'>Error: {str(e)}</div>"
            explain_html = "Prediction failed — check console logs for details."
            return annotated_image, "Prediction failed", bar_str, explain_html


# --- 4. LAUNCH THE APP ---
# This line starts the web server.
iface.launch()
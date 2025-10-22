# app.py

import gradio as gr
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import os
import random

# Optimize ONNX Runtime for CPU multi-threading
num_threads = os.cpu_count() or 1
print(f"Using device: CPU (threads={num_threads})")

# --- 1. LOAD THE MODEL (validated & fail-fast) ---
# Prefer an exported ONNX model when present, otherwise fall back to .pt.
# Fail fast with clear logs if no model artifact is present in the image.
model_path = None
searched = ['model/best.onnx', 'model/best.pt', 'best.onnx']
for p in searched:
    if os.path.exists(p):
        model_path = p
        break

if model_path is None:
    print("ERROR: No model artifact found in container. Expected one of: model/best.onnx, model/best.pt, or best.onnx")
    print("Searched locations:", ", ".join(searched))
    print("Please ensure the trained model is added to the Docker build context and copied into the image (or implement a runtime download).")
    import sys
    sys.exit(1)

print(f"Using model artifact at: {model_path}")
# Note: model was trained with a YOLOv11 architecture (training script in model/training_script.py)
session = None
model = None
if model_path.endswith('.onnx'):
    try:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        sess_options.log_severity_level = 3
        session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"ERROR: Failed to load ONNX model at '{model_path}': {e}")
        import traceback, sys
        traceback.print_exc()
        sys.exit(1)
else:
    # Fallback: lazy-import ultralytics only if needed
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
    except Exception as e:
        print(f"ERROR: Failed to load PyTorch model at '{model_path}': {e}")
        import traceback, sys
        traceback.print_exc()
        sys.exit(1)

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
            # If we have an ONNX session, run ONNX inference; otherwise use ultralytics model.
            if session is not None:
                # Preprocess: letterbox to 640x640 (model expected size from training script)
                def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
                    h, w = im.shape[:2]
                    r = min(new_shape[0] / h, new_shape[1] / w)
                    nh, nw = int(round(h * r)), int(round(w * r))
                    resized = cv2.resize(im, (nw, nh))
                    canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
                    dy = (new_shape[0] - nh) // 2
                    dx = (new_shape[1] - nw) // 2
                    canvas[dy:dy + nh, dx:dx + nw, :] = resized
                    return canvas, r, dx, dy
                img_resized, r, dx, dy = letterbox(image)
                img_input = img_resized.astype(np.float32) / 255.0
                # Convert HWC -> NCHW
                img_input = np.transpose(img_input, (2, 0, 1))[np.newaxis, ...]
                input_name = session.get_inputs()[0].name
                outputs = session.run(None, {input_name: img_input})
                out = outputs[0]
                # Interpret outputs for common YOLO ONNX exports:
                # Expecting shape (1, N, D) where D >= 6 (x1,y1,x2,y2,obj_conf,[class_probs...]) or (N,6)
                results = []
                if out is None:
                    results = []
                else:
                    arr = np.array(out)
                    if arr.ndim == 3:
                        dets = arr[0]
                    elif arr.ndim == 2:
                        dets = arr
                    else:
                        dets = arr.reshape(-1, arr.shape[-1])
                    # When dims > 6, compute class id via argmax of class probs
                    if dets.size == 0:
                        results = []
                    else:
                        D = dets.shape[1]
                        if D >= 6:
                            if D > 6:
                                class_probs = dets[:, 5:]
                                class_ids = np.argmax(class_probs, axis=1)
                                class_scores = class_probs[np.arange(len(class_ids)), class_ids]
                                confs = dets[:, 4] * class_scores
                            else:
                                class_ids = dets[:, 5].astype(int)
                                confs = dets[:, 4]
                            boxes = dets[:, :4].copy()
                            # Undo letterbox padding/scale to original image coords
                            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / r
                            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / r
                            # Build ultralytics-like results for downstream code reuse
                            results = []
                            for i in range(len(boxes)):
                                b = boxes[i].tolist()
                                box = type("box", (), {})()
                                box.xyxy = np.array([b])
                                box.cls = np.array([int(class_ids[i])])
                                box.conf = np.array([float(confs[i])])
                                results.append(type("r", (), {"boxes": [box]}))
                        else:
                            results = []
                inference_end = time.time()
                inference_time = inference_end - start_time
                print(f"Inference time: {inference_time:.2f}s")
                annotated_image = image.copy()  # Use copy to avoid modifying input
            else:
                results = model(image)
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
# Start the web server. Add explicit startup logs and keep-alive fallback so
# platform health checks can see the process and we can debug if Gradio exits.
print(f"Starting Gradio on 0.0.0.0:{os.environ.get('PORT', 8080)} ...")
try:
    iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)), enable_queue=True)
    # If iface.launch blocks normally, code below won't run. If it returns, log and keep container alive.
    print("Gradio launch() returned — entering keep-alive loop for debugging (container will remain up).")
except Exception as e:
    print(f"Gradio raised an exception during launch: {e}")
    import traceback
    traceback.print_exc()

# Keep the process alive after launch returns so Fly health checks can succeed while we debug.
import time
print("Entering fallback keep-alive loop (sleep). Press Ctrl+C to exit.")
while True:
    time.sleep(60)
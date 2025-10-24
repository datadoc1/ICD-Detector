# ICD Detector (Space)
---
title: ICD Detector
emoji: 🫀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---
**ICD Detector: Spot Implanted Cardiac Devices in Chest X-Rays.**

**Overview (short):**
Built and hosted by Linh Nguyen, Keola Ching, and Dr. Kal Clark this demo uses a YOLOv11-based model trained from a Roboflow export to explore automated detection of implantable cardioverter‑defibrillators (ICDs) on chest X-rays. Upload a CXR screenshot and the demo will attempt to identify device(s), display bounding boxes, and show a confidence bar with a short interpretation.

Standard project layout (cleaned):
- Model artifact: model/best.pt
- Training script: model/training_script.py
- Dataset layout: model/data/train, model/data/val, model/data/test
- Dataset config (YAML): model/data.yaml

---

## Introduction
This repo powers an interactive Gradio demo that detects and highlights implanted cardiac device hardware on chest radiographs. It is a research/education prototype — not clinical software — built from a Roboflow export and trained by Linh Nguyen.

Key credits:
- Linh Nguyen — dataset labeling and model training (primary author of dataset and training).
- Keola Ching — model adaptation, hosting the demo, and integrating the Gradio UI.
- Dr. Kal L. Clark — project advisor.

Dataset & source:
- Roboflow dataset: https://universe.roboflow.com/nighthawklvn/icd-xray-image-recognition
- Training and project files in this repo show how the demo was assembled from the Roboflow export and YOLOv11 training.

The dataset for the prototype contains labelled CXR images covering devices from Biotronik, Boston Scientific, Medtronic, and St. Jude.

---

## How it works
- Upload a chest X-ray (screenshot or JPEG/PNG).
- The app runs a YOLOv11 model (artifact: `model/best.pt`) to detect device bounding boxes and predict the most likely device class.
- The UI displays the annotated image, a colored confidence bar (green/yellow/red), and a short interpretation text to help non-expert users understand results.

Notes for reproducing training:
- Training script: `model/training_script.py` (calls YOLOv11, points to `model/data.yaml` / `model/data/` folders).
- Dataset layout should follow: `model/data/train`, `model/data/val`, `model/data/test`. Update `model/data.yaml` to match these paths before training.

---

## Limitations & Fun Facts
- **Not a medical device.** This is for educational/entertainment purposes only. Always consult clinical experts and official MRI safety resources for patient care.
- Trained on a very small dataset (98 images) — performance is limited and overfitting / misclassification is possible.
- Pacemakers and ICDs can look very similar radiographically; the model may label pacemakers as ICDs when shock coils are not visible.
- Fun fact: ICDs often have thicker, coiled defibrillator leads (shock coils) — radiologists look for these to confirm ICDs.

---

## Device Table (transparency)
Include the project's device table (18 device types) here for transparency. Recommended: upload an image `images/device_table.png` or convert your table to markdown below.

![Device table placeholder](images/device_table.png)

If you prefer a markdown table, paste it below.

---

## Links & Resources
- Roboflow dataset (replace with your dataset URL): https://app.roboflow.com/
- Training code (replace with your gist or repo): https://gist.github.com/your-training-code
- MRI safety reference: https://www.mrisafety.com
- Gradio docs: https://gradio.app

---

## Future ideas
- Add pacemaker vs ICD differentiation and larger dataset for better accuracy.
- Expose model metadata (which class came from which manufacturer/model).
- Add an option to toggle bounding-box overlays and download annotated images.
- Mobile-friendly layout improvements and example gallery to demonstrate typical correct/incorrect outputs.

---

## Contributing / Notes
- To reproduce training, include dataset export and training script links above.
- This Space is a prototype and meant to spark discussion — please open issues or PRs with improvements.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

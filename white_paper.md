
# ICD Detector: Revolutionizing MRI Safety & Throughput

From Bottleneck to High-Throughput Triage using Computer Vision

## Executive Summary
The ICD Detector project is a strategic initiative designed to solve a critical operational failure in modern radiology: the inefficiency of clearing patients with Cardiac Implantable Electronic Devices (CIEDs) for MRI scans. While currently functioning as a technical Proof‑of‑Concept (POC) utilizing state‑of‑the‑art YOLOv11 computer vision to detect devices in Chest X‑Rays (CXRs), the roadmap extends beyond simple detection. The vision is to build an end‑to‑end AI‑driven triage engine. By automating the identification of devices and mapping them to safety protocols, we aim to unlock significant "uncaptured revenue," reduce patient wait times from weeks to minutes, and streamline the pre‑MRI safety workflow.

## The Market Challenge: Rising Volume, Rising Friction
Magnetic Resonance Imaging (MRI) utilization is accelerating globally. In the U.S. alone, major provider networks reported a 14% increase in scan volume in Q3 2023, with some medical centers seeing nearly 30% growth over a two‑year period.

However, as volume grows, so does the complexity of the patient population. Approximately 75–80% of pacemaker recipients will require an MRI during their lifetime. Currently, these patients face a "safety bottleneck":

- 50x Lower Referral Rate: Patients with CIEDs are significantly less likely to receive necessary imaging due to complex contraindication protocols.
- Operational Drag: Clearing a CIED patient often requires coordination between radiology, cardiology, and device manufacturers, leading to delays exceeding one month.

## The Economic Impact: The Million‑Dollar Problem
The friction in current workflows results in lost revenue. A 2022 study on academic radiology practices indicated that appointment no‑shows and scheduling inefficiencies—often caused by safety clearance delays—can cost a single site nearly $1 million annually in uncaptured revenue. High‑value exams (e.g., Brain, Spine MRIs) account for over $100,000 of this loss per site. Every delayed or cancelled scan represents an unrecoverable "slot" in scanner utilization time.

## Technical Solution: The YOLOv11 Advantage
The core of our solution is an automated detection pipeline that acts as the "gatekeeper" for MRI safety.

1. Current Architecture (The POC)  
   We have deployed a prototype using YOLOv11 (You Only Look Once, v11), the industry standard for real‑time object detection released in late 2024.  
   - Input: Chest X‑Rays (CXRs).  
   - Output: Bounding‑box localization, device classification (ICD vs. Pacemaker), and confidence scoring.  
   - Deployment: Web‑based interface allowing for instant upload and analysis.

2. Data Strategy & Scalability  
   While the prototype was validated on a boutique dataset to prove feasibility, our roadmap for production readiness utilizes massive, clinically validated datasets to ensure diagnostic‑grade accuracy.  
   - Targeted Training: Leveraging the PhysioNet specialized dataset (2,321 images specifically classified for cardiac devices across 27 models and 4 manufacturers).  
   - Generalizability: Pre‑training on MIMIC‑CXR (377,110 images) to ensure the model understands diverse chest anatomies and radiographic variances.

3. Competitive Landscape  
   While academic studies (e.g., 2019 neural network research on 1,400 images) have proven that AI can detect devices, our value proposition differs. We are not just building a detector; we are building a Workflow Integration System.  
   - Competitors: Focus on static identification.  
   - ICD Detector: Focuses on dynamic triage—linking the detection immediately to manufacturer safety databases (e.g., MRISafety.com) to output a "Go/No‑Go" decision support signal.

## Operational Vision: The "Zero‑Click" Workflow
We aim to move from reactive, manual checks to proactive, automated screening.

Workflow Stage | Current Standard | Future State (ICD Detector)
--- | --- | ---
Intake | Patient declares device; tech manually searches records. | Automated: AI scans intake CXR and flags device type instantly.
Verification | Tech calls cardiology or checks device card (if available). | Integration: System retrieves device make/model and pulls contraindications.
Decision | Radiologist/Cardiologist consult (hours/days). | Triage: Instant recommendation (e.g., "Conditional Safe: Switch to Mode B").
Throughput | High cancellation rate; unused scanner time. | Optimized: 20–50% improvement in throughput via automated scheduling.

Evidence supports this efficiency leap. AI‑powered worklists in similar domains have demonstrated a 34% improvement in study distribution, and AI reporting tools have boosted documentation efficiency by 15.5%.

## Project Team & Roots
This project bridges the gap between clinical reality and software engineering.

- Linh Nguyen (Project Lead & Domain Expert): Former MRI Technologist with firsthand experience in the safety bottlenecks of CIED imaging. Led dataset labeling and model training.  
- Keola Ching (Technical Lead): Full‑stack engineer responsible for model adaptation, web UI, and deployment architecture.  
- Dr. Kal L. Clark (Advisor): Providing clinical oversight and strategic guidance.

## Conclusion & Next Steps
The ICD Detector is not just a tool for identifying hardware; it is a mechanism for unlocking MRI capacity. By automating the most friction‑heavy part of the safety process, we ensure that patients get the scans they need, and hospitals capture the revenue they are currently losing.

For a live demonstration of the detection capability, upload a CXR to our prototype interface.

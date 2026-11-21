# A Dual-Module System Combining Sign Language Recognition and Hand-Tracking Interaction

**Author:** Yi-Hsuan Wu  
**Affiliation:** Department of Electrical Engineering, Chang Gung University  
**Conference:** Taiwan Society of Biomedical Engineering (TSBME) Annual Symposium  

---

## 📌 Project Architecture
This project implements an integrated dual-module platform:
1. **Module 1 (Sign Language Recognition):**
   - **Feature Extraction:** 21 hand landmarks (63-D coordinate vectors) extracted per frame via Google MediaPipe Hands.
   - **Temporal Modeling:** 60-frame sequential tensors processed by a **Conv1D + LSTM** deep learning architecture.
   - **Classification:** 64 sign gesture categories achieving **~87% Top-1 accuracy**.
2. **Module 2 (Real-Time Hand Tracking & Interaction):**
   - Leap Motion sensor capturing 3D hand skeleton dynamics up to 120 Hz.
   - Low-latency Python socket streaming JSON payloads over TCP to Unity C# scripts.
   - End-to-end interactive manipulation latency <= 100 ms.

---

## 📂 Repository Structure
```
├── data_extraction.py       # Landmark extraction, wrist-centering, and 60-frame sequence alignment
├── train_model.py           # Conv1D + LSTM model definition, callbacks, and training script
├── real_time_inference.py   # Live webcam landmark extraction and real-time prediction overlay
├── unity_leap_bridge.py     # Leap Motion 3D skeleton streaming to Unity via JSON/TCP
├── sample_dataset.csv       # Preprocessed 60-frame x 63-D normalized landmark sample
├── requirements.txt         # Environment dependencies
└── README.md                # Documentation and execution guide
```

---

## 🚀 Quick Start
```bash
pip install -r requirements.txt
python train_model.py
```

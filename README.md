# Mexican Sign Language Real-Time Interpreter

Real-time translation system from Mexican Sign Language (LSM) to text, developed as a Mechatronics Engineering capstone project at Instituto Politécnico Nacional (IPN — UPIITA).

The system detects hand gestures via webcam, identifies the corresponding LSM sign using a custom multi-channel CNN, and displays the translated word in real time through a web interface.

---

## Demo

| Hand detection (YOLOv7) | Sign classification (CNN) |
|:-:|:-:|
| Detects and crops the hand region from the webcam feed | Classifies the cropped region into one of 15 LSM signs |

---

## Supported signs

`Hola` · `Como` · `Estar` · `Bien` · `Gracias` · `Que` · `Hacer` · `Tu` · `Tambien` · `Comer` · `Trabajar` · `Mal` · `Si` · `No` · `Adios`

---

## System architecture

```
Webcam
  │
  ▼
YOLOv7 hand detector  ──►  Cropped hand region (400×400 px)
                                      │
                                      ▼
                         Multi-channel CNN classifier
                         ┌──────────────────────────┐
                         │  Branch R  (red channel)  │
                         │  Branch G  (green channel)│──► Concatenate ──► Dense ──► Softmax
                         │  Branch B  (blue channel) │
                         │  Branch BN (grayscale)    │
                         └──────────────────────────┘
                                      │
                                      ▼
                              Predicted LSM sign
                                      │
                                      ▼
                           Web interface (HTML + TF.js)
```

---

## Repository structure

```
mexican-sign-language-interpreter/
│
├── detection/
│   └── real_time_detection.py   # YOLOv7-based hand detection (modified)
│
├── training/
│   └── train_cnn.py             # Multi-channel CNN training pipeline
│
├── scripts/
│   └── run_detection.py         # Launcher script for the detection module
│
├── web/
│   └── index.html               # Web interface — displays real-time predictions
│
└── README.md
```

---

## Tech stack

| Area | Tools |
|---|---|
| Hand detection | YOLOv7, PyTorch, OpenCV |
| CNN training | TensorFlow / Keras, NumPy, scikit-learn |
| Web interface | HTML, Bootstrap 5, TensorFlow.js |
| Backend server | Python (Flask) |
| GPU acceleration | CUDA |
| Data format | JSON (model serialization: `.h5` → `model.json`) |

---

## Dataset

- **3,750 labeled images** (200×200 px) collected from 30+ contributors
- Images cropped automatically from YOLOv7 bounding-box detections
- Balanced across 15 LSM sign classes using stratified splitting
- Augmented with parallel RGB and grayscale channel extraction

---

## Model

Custom multi-channel CNN with four parallel branches — one per color channel (R, G, B, Grayscale). Each branch applies three Conv2D + MaxPooling blocks, then flattens. The four feature vectors are concatenated and passed through Dense + Dropout layers before a 16-class Softmax output.

**Evaluation metrics:** F1-score, Precision, Recall, Specificity, Confusion Matrix

---

## Getting started

### 1. Clone the repository
```bash
git clone https://github.com/FrankAlvr/mexican-sign-language-interpreter.git
cd mexican-sign-language-interpreter
```

### 2. Install dependencies
```bash
pip install torch torchvision opencv-python tensorflow numpy scikit-learn matplotlib
```

### 3. Run detection
Update the paths in `scripts/run_detection.py`, then:
```bash
python scripts/run_detection.py
```

### 4. Open the web interface
Open `web/index.html` in a browser while the Flask server is running.

---

## Development methodology

Structured following **VDI 2206** systems engineering lifecycle:
- Stakeholder needs capture
- Functional and non-functional requirement derivation
- Subsystem decomposition and interface definition
- Quantitative verification and validation (V&V)

---

## Author

**Francisco Javier Alvarado Angeles**  
Mechatronics Engineer — IPN UPIITA  
[linkedin.com/in/francisco1700](https://linkedin.com/in/francisco1700) · [github.com/FrankAlvr](https://github.com/FrankAlvr)

**Advisors:** Dr. Rene Luna García · Dra. Obdulia Lagunas Pichardo

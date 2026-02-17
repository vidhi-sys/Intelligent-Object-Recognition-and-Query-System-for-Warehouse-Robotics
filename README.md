

# 🤖 WarehouseMind: Intelligent Vision + Learning + Retrieval for Robotics

> A miniature AI brain for warehouse robots — combining perception, reasoning, and knowledge retrieval in one unified system.

---

## 🧠 What Is This?

**WarehouseMind** simulates how a smart warehouse robot would:

* 👀 See objects through a camera
* 🏷 Classify them by handling category
* 📚 Retrieve correct operational documentation
* 💬 Answer safety and handling queries

Instead of building isolated ML scripts, this project connects:

> **Computer Vision + Deep Learning + Retrieval-Augmented Generation (RAG)**

into one cohesive AI pipeline.

---

# 🏭 Why This Project?

Modern warehouse robots don’t just need to “see” objects.

They must:

* Identify what kind of object it is
* Understand how to handle it
* Follow safety protocols
* Retrieve technical documentation instantly

This project replicates that intelligence stack at a smaller scale.

---

# 🧩 System Components

---

## 1️⃣ Vision Module — Object Detection (OpenCV)

📸 Simulates robot perception.

* HSV color segmentation
* Contour detection
* Bounding box extraction
* Object dimension estimation
* Center coordinate calculation

📦 Detects packages, boxes, and colored objects in test images.

**Output includes:**

* Annotated image with bounding boxes
* Object size & position
* Segmentation mask

---
<img width="1680" height="452" alt="image" src="https://github.com/user-attachments/assets/3642bcf8-fb1f-4da8-be8d-d6b846a39b5f" />

## 2️⃣ Machine Learning Module — Object Classification

🧠 A lightweight CNN trained on CIFAR-10 (remapped into warehouse categories).

### Categories:

* **Fragile**
* **Heavy**
* **Hazardous**

Why CIFAR-10?

* Small
* Easy to download
* Memory efficient
* Good for quick experimentation

### Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Model weights are saved for reuse.

Designed intentionally lightweight to prevent RAM crashes.

---
<img width="514" height="470" alt="image" src="https://github.com/user-attachments/assets/0387b4c8-85be-4c2a-a50c-bd6865290f5c" />

## 3️⃣ RAG Module — Robotics Knowledge Retrieval

📚 A synthetic robotics documentation knowledge base (10–15 documents).

Includes:

* Handling instructions
* Safety procedures
* Equipment specifications
* Troubleshooting guides

### Pipeline:

1. Chunk documents
2. Generate embeddings
3. Store in FAISS vector database
4. Retrieve top-k relevant chunks
5. Generate grounded response

---
<img width="1672" height="260" alt="image" src="https://github.com/user-attachments/assets/4a30de51-2d41-47d0-94a1-32bfeee5b70c" />

# 🔍 Example Queries the System Handles

* “How should the robot handle fragile items?”
* “What is the maximum weight capacity of the gripper arm?”
* “What safety checks are required before moving hazardous materials?”
* “What should be done if grip failure occurs?”

The system retrieves relevant documentation before generating the answer — reducing hallucination risk.

---

# 🧠 Architecture Overview

```
Camera Input
     ↓
OpenCV Detection
     ↓
CNN Classifier
     ↓
Warehouse Category
     ↓
RAG Knowledge Retrieval
     ↓
Grounded AI Response
```

This simulates a **perception → reasoning → action support** pipeline.

---

# ⚙️ Installation

```bash
pip install torch torchvision
pip install opencv-python numpy
pip install scikit-learn matplotlib seaborn
pip install sentence-transformers faiss-cpu
```

---

# ▶️ How to Run

### 🔹 Vision Module

```bash
python part1_opencv.py
```

### 🔹 Machine Learning Training

```bash
python warehouse_classifier_light.py
```

### 🔹 RAG System Demo

```bash
python warehouse_rag.py
```

---

# 📊 Sample Performance (Typical)

* Accuracy: ~75–85%
* Balanced precision & recall
* Confusion matrix saved as PNG
* Lightweight model avoids RAM crashes

---

# ⚠️ Limitations

* Visual classification cannot truly detect “weight” or “hazardous” nature — only image patterns.
* CIFAR-10 remapping is artificial.
* Color-based segmentation is lighting sensitive.
* RAG system uses synthetic documentation.

---

# 🚀 Future Improvements

* YOLO-based object detection
* Real warehouse dataset
* Sensor fusion (vision + weight sensors)
* LLM-powered natural answer generation
* Deployment on edge devices (Jetson Nano / Raspberry Pi)

---

# 🎥 Ideal Demo Flow (5 Minutes)

1. Show object detection
2. Show classification results
3. Run example queries
4. Display retrieved documentation
5. Show grounded response

---
https://drive.google.com/file/d/1YA-kw47Rf4dG6b-vcWJnJqeM3LLqcPUB/view?usp=sharing

# 🏁 What This Project Demonstrates

✔ Applied Computer Vision
✔ Deep Learning under memory constraints
✔ Vector databases (FAISS)
✔ Retrieval-Augmented Generation
✔ End-to-end AI system integration

This is not just a model — it’s a **mini AI robotics framework**.

---

# 👩‍💻 Author

Vidhi
CSE – AIML
VIT Bhopal University









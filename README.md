Raspberry Pi Edge Face Recognition System
Overview
This project implements a real‑time face recognition pipeline on a Raspberry Pi 5 using an edge‑AI approach. The system captures video from a camera module, detects faces using a deep learning model, generates embeddings, and performs fast identity matching using FAISS.

The goal of the project is to demonstrate a portable facial recognition device suitable for mobile surveillance or patrol scenarios, where inference is performed entirely on-device without cloud dependency.

Features
Real‑time face detection and recognition

Edge deployment on Raspberry Pi 5

Efficient similarity search using FAISS

Deep learning embeddings generated with InsightFace (ArcFace / Buffalo_L)

Video capture and processing using OpenCV / Picamera2

Optimized for CPU inference

System Architecture
Pipeline:

Camera Input
      ↓
Frame Capture (OpenCV / Picamera2)
      ↓
Face Detection (InsightFace)
      ↓
Embedding Extraction
      ↓
FAISS Similarity Search
      ↓
Identity Recognition
      ↓
Annotated Video Output
Technologies Used
Hardware
Raspberry Pi 5

Raspberry Pi Camera Module

Active cooling module

Software
Python

OpenCV

InsightFace

FAISS

NumPy

Pandas

ONNX Runtime


Install dependencies:

pip install opencv-python insightface faiss-cpu numpy pandas onnxruntime
Usage
Run the main face recognition pipeline:

python face_recognition_pipeline.py
The system will:

Capture frames from the camera

Detect faces

Generate embeddings

Compare embeddings with the database

Display recognized identities

Dataset / Face Database
Face embeddings are stored in a CSV file containing:

512‑dimensional embedding vectors

Associated identity labels

Example format:

label,0,1,2,...,511
person_1,0.12,0.43,...,0.55
person_2,0.09,0.33,...,0.61
Repository Structure
project-root
│
├── face_recognition_pipeline.py   # Core ML pipeline
├── face_embeddings.csv            # Face embedding database
├── README.md
└── requirements.txt
Limitations
Performance depends on CPU availability of the Raspberry Pi.

Recognition accuracy depends on lighting conditions and database quality.

Large embedding databases may increase search latency.

Future Improvements
Hardware acceleration (NPU / GPU)

Multi-camera support

Improved tracking to reduce redundant inference

Mobile interface for field deployment

Author Rakshith R R

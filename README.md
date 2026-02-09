# Real-Time Football Video Analytics using YOLOv8

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Object_Detection-green) ![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-white) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red)

An end-to-end computer vision system for **football video analytics**, designed to process match footage frame-by-frame and generate continuous, annotated outputs. The pipeline integrates object detection, tracking, team assignment, possession analysis, and motion estimation using modern deep learning and vision techniques.

This project emphasizes **real-time inference pipelines**, modular system design, and practical ML engineering.

---

## Table of Contents
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Core Technologies](#-core-technologies)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Notes on Excluded Files](#-notes-on-excluded-files)

---

## Key Features

The system takes a football match video as input and produces an annotated output video containing:

- **Object Detection**: Detects players, referees, and the ball.
- **Multi-Object Tracking**: Assigns consistent IDs across frames (using ByteTrack logic).
- **Team Assignment**: Classifies players into teams using K-Means clustering on jersey colors.
- **Ball Possession**: Dynamically calculates which player has possession based on proximity.
- **Camera Motion Compensation**: Uses Optical Flow to adjust calculations for camera movement.
- **Speed & Distance**: Estimates physical speed and distance covered by players.

### "Real-Time" Processing Philosophy
Although the input is a video file, the system processes frames **sequentially (one at a time)**. This mirrors real-world production environments where frames arrive sequentially, preventing batch-style or post-hoc processing.

---

## System Architecture

The pipeline is structured as a sequence of modular components:

1.  **Frame Ingestion**: Read video frames sequentially.
2.  **Object Detection**: Detect players and ball using **YOLOv8**.
3.  **Multi-Object Tracking**: Maintain consistent IDs across frames.
4.  **Team Assignment**: Classify players into teams using jersey color clustering.
5.  **Player–Ball Assignment**: Track possession dynamically.
6.  **Camera Motion Estimation**: Compensate for camera panning/zooming.
7.  **Speed & Distance Estimation**: Transform perspective to measuring units.
8.  **Annotation & Visualization**: Generate analytics-rich output video.

---

## Core Technologies

* **YOLOv8 (Ultralytics)**: State-of-the-art object detection.
* **ByteTrack**: Robust multi-object tracking.
* **OpenCV**: Video processing and optical flow.
* **PyTorch**: Deep learning inference backend.
* **Scikit-Learn**: K-Means clustering for color segmentation.
* **Pandas/Numpy**: Data manipulation.

---

## Project Structure

```text
.
├── camera_movement_estimator/   # Optical flow logic for camera stabilization
│   ├── __init__.py
│   └── camera_movement_estimator.py
├── player_ball_assigner/        # Logic to link ball detection to player ID
│   ├── __init__.py
│   └── player_ball_assigner.py
├── speed_and_distance_estimator/# Perspective transformation & metric calc
│   ├── __init__.py
│   └── speed_and_distance_estimator.py
├── team_assigner/               # K-Means clustering for jersey colors
│   ├── __init__.py
│   └── team_assigner.py
├── trackers/                    # YOLO + ByteTrack wrapper
│   ├── __init__.py
│   └── tracker.py
├── utils/                       # Bounding box and video I/O utilities
│   ├── __init__.py
│   ├── bbox_utils.py
│   └── video_utils.py
├── view_transformer/            # Perspective transformation logic
│   ├── __init__.py
│   └── view_transformer.py
├── development_and_analysis/    # Notebooks for color analysis & prototyping
├── stubs/                       # Serialized pickle files (mock data)
├── main.py                      # Main pipeline execution script
├── yolo_inference.py            # Inference testing script
├── requirements.txt             # Dependencies
└── README.md

Setup & Installation
1️⃣ Clone the Repository
```bash
git clone [https://github.com/dhaya56/real-time-football-video-analytics.git](https://github.com/dhaya56/real-time-football-video-analytics.git)
cd real-time-football-video-analytics

2️⃣ Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies
```bash
pip install -r requirements.txt

4️⃣ Download Model Weights
This repository does not include trained YOLO model weights to keep the codebase lightweight.

Download YOLOv8 weights (e.g., yolov8x.pt or yolov8l.pt) from the official Ultralytics repository.

Place the .pt file in the root directory.

Usage
Prepare Input Video: Place your football match video (e.g., match.mp4) in an input_videos/ folder.

Run the Pipeline:
```bash
python main.py --input input_videos/match.mp4 --output output_videos/annotated_match.mp4
Note: If you run main.py without arguments, ensure the file paths inside the script match your local directory structure.

Notes on Excluded Files
To maintain clarity and repository size constraints, the following are excluded from the repo:

    Trained Model Weights (.pt)

    Input/Output Videos

    Training Artifacts & Logs
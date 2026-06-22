# 📦 Vision-Based Inventory Management System

A real-time, AI-powered inventory tracking application built entirely with Python and OpenCV. This system uses a custom-trained YOLOv8 neural network to detect objects via webcam and automatically allocates them to a virtual 4x4 smart-shelf interface, complete with a real-time analytics dashboard.

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-UI%20%26%20Vision-red)

## ✨ Features

* **Real-Time AI Detection:** Utilizes a custom YOLOv8 model (exported to ONNX) to accurately identify objects in a live webcam feed.
* **Custom OpenCV GUI:** The entire user interface—including the 4x4 shelf grid, video feed overlay, and circular progress gauges—is mathematically drawn from scratch using OpenCV (no external GUI libraries like Tkinter or PyQt).
* **Smart Shelf Allocation:** Automatically routes detected items (Phones, Watches, Keys, Toothpaste) to their designated rows and calculates shelf capacity.
* **Class-Specific Confidence Thresholds:** Uses dynamically tuned strictness levels for different objects (e.g., highly strict for phones, forgiving for keys) to eliminate false positives.
* **Debounce & Cooldown Logic:** Prevents duplicate counting of the same object by implementing a frame-based cooldown timer during the detection phase.
* **Real-Time Dashboard:** Displays overall capacity via a dynamic circular gauge and a tabular breakdown of individual shelf statuses.

## 🛠️ Tech Stack

* **Language:** Python 3
* **Computer Vision & GUI:** OpenCV (`cv2`)
* **Matrix Math:** NumPy (`numpy`)
* **AI Model:** YOLOv8 (Running via OpenCV's `dnn` module)

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/vision_inventory.git](https://github.com/YOUR_USERNAME/vision_inventory.git)
   cd vision_inventory

   pip install opencv-python numpy
2. **Add your AI Model:**
   Ensure you have your custom-trained YOLOv8 ONNX model.Rename it to best.onnx.Place it directly into the root vision_inventory directory. (Note: This file should be ignored by .gitignore due to size constraints on GitHub).

2. 💻 **Usage:**
   Run the main application script:
   ```bash
   python main.py

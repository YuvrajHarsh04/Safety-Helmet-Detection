Safety Helmet Detection

This repository contains a machine learning-based safety helmet detection system designed for construction sites. The project is developed using Python, Jupyter Notebook, and Streamlit to provide an intuitive and interactive interface for real-time detection and analysis.

Features

Real-time Detection: Processes video streams or images to identify workers with or without helmets.

Interactive Interface: Streamlit-based user interface for easy interaction and visualization.

High Accuracy: Trained with a diverse dataset to ensure robustness in varying lighting and environmental conditions.

Customizable: Easily adaptable for different use cases or additional safety gear detection.

Deployment Ready: Can be integrated with edge devices, surveillance systems, or web platforms.

Getting Started

Prerequisites

Python 3.8+

Libraries: TensorFlow/PyTorch, OpenCV, NumPy, Streamlit, etc. (see requirements.txt)

Jupyter Notebook for development and experimentation.

A GPU-enabled system for training or high-performance inference (optional but recommended).

Installation

Clone the repository:

git clone https://github.com/yourusername/safety-helmet-detection.git
cd safety-helmet-detection

Install dependencies:

pip install -r requirements.txt

Prepare your dataset:

Place images in data/images and labels in data/labels.

Run the Jupyter Notebook for training:

jupyter notebook
# Open and run `train_model.ipynb`

Launch the Streamlit app for detection:

streamlit run app.py

How It Works

Data Preprocessing: Images are resized and normalized, and corresponding labels are processed.

Model Training: Utilizes a Convolutional Neural Network (CNN) or YOLO-based architecture.

Helmet Classification: Detects and classifies helmeted and non-helmeted workers in the input.

Streamlit Interface: Provides an interactive interface for users to upload images/videos and view detection results.

Screenshots

![Screenshot](imgs/Screenshot%202024-12-19%20at%2010.29.29%E2%80%AFPM.png)

![Screenshot](imgs/Screenshot%202024-12-19%20at%2010.41.07%E2%80%AFPM.png)


Use Cases

Construction site safety monitoring.

Real-time compliance checks.

Incident prevention systems.

# Face Emotion Recognition 🎭

A simple deep learning model to detect human emotions from facial expressions using Convolutional Neural Networks (CNN).

## 🔍 Overview
This project aims to classify facial expressions into different emotions such as:
- Happy
- Sad
- Angry
- Surprise
- Neutral
- Fear
- Disgust

The model was trained using the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) and implemented with TensorFlow/Keras.

## 🧠 Technologies Used
- Python
- OpenCV
- TensorFlow / Keras
- NumPy, Matplotlib
- Flask (for web demo)

## 📁 Project Structure

- `train.py` – Trains the CNN model on FER-2013.
- `predict.py` – Loads the model and predicts emotion from input images.
- `app.py` – (Optional) Flask app for real-time webcam inference.
- `model/` – Contains trained model weights.
- `static/` – Example input/output images.
- `requirements.txt` – List of dependencies.

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/rNEGI1729/Face-emotion-recognition.git
  
   cd Face-emotion-recognition

# ASL Sign Language Detector

This project aims to detect American Sign Language (ASL) letters using computer vision techniques and machine learning. The detector is capable of converting ASL signs into both spoken language and text format.

## Features

- **ASL Sign Detection**: Detects and interprets ASL gestures using a trained machine learning model.
- **Speech Output**: Converts detected signs into spoken language.
- **Text Output**: Provides text output corresponding to detected signs.
- **Model Persistence**: Uses pickle for saving and loading trained models.

## Libraries Used

- **pickle**: For serializing and deserializing Python objects, used here for saving the trained model.
- **scikit-learn (sklearn)**:
  - `RandomForestClassifier`: Utilized for training the ASL gesture recognition model.
  - `train_test_split`: For splitting the dataset into training and testing sets.
  - `accuracy_score`: Measures the accuracy of the trained model.
- **numpy (np)**: Essential for numerical operations and array handling.
- **mediapipe**: Provides solutions for perception tasks, including hand tracking and pose estimation.
- **cv2 (OpenCV)**: For computer vision tasks, such as image and video processing.
- **os**: Allows interaction with the operating system, used for file and directory operations.
- **matplotlib.pyplot as plt**: Used for visualizing data and results.

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/your_username/ASL-Sign-Language-Detector.git
    cd ASL-Sign-Language-Detector
    ```
<br>
2. Install dependencies:

    ```
    pip install -r requirements.txt
    ```
<br>
3. Download the ASL Alphabet dataset:

  - Download the dataset from [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/grassknoted/asl-alphabet)
  - Extract the dataset files into a directory named ```Data```.

 

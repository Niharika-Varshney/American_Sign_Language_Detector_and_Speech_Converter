# ASL Sign Language Detector and Speech Converter

## Overview
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
## Output
![IMG_2895](https://github.com/Niharika-Varshney/Sign_language_Detector-Speech_Covertor/assets/118551235/ed501e48-31d8-440c-8cd8-9f5cfad4d416)
![IMG_2901](https://github.com/Niharika-Varshney/Sign_language_Detector-Speech_Covertor/assets/118551235/b6657c3e-5600-4080-86a1-2931ffd4e026)
![IMG_2894](https://github.com/Niharika-Varshney/Sign_language_Detector-Speech_Covertor/assets/118551235/79d1a44b-942b-4dbc-8fae-1857518b22d0)
![IMG_2896](https://github.com/Niharika-Varshney/Sign_language_Detector-Speech_Covertor/assets/118551235/40e2f822-4ce5-4204-b4e0-a90fd9ca3f48)

## Installation
1. Clone the repository:

    ```
    git clone https://github.com/Niharika-Varshney/Sign_language_Detector-Speech_Covertor
    cd Sign_language_Detector-Speech_Covertor
    ```
<br>

2. Download the ASL Alphabet dataset:

  - Download the dataset from [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/grassknoted/asl-alphabet)
  - Extract the dataset files into a directory named ```Data```.


## Files Included:
- ```create_datasetimage.py```: To create and add images of ASL signs to the ```Data``` directory or you can save [Kaggle ASL Alphabet Dataset] in ```Data``` directory.
- ```dataset_process```: To process the dataset and save it.
- ```classifier.py```:To train the model on processed datset to detect the ASL letters and saving it.
- ```testing_dataset```:Testing the model using a webcam.

 

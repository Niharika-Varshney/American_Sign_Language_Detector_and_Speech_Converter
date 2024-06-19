import mediapipe as mp
import cv2 as cv
import os
import matplotlib.pyplot as plt
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the hands object with static_image_mode=True for processing images
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
data = []
labels = []
datadir = 'Data'
for dir_ in os.listdir(datadir):
    for img_path in os.listdir(os.path.join(datadir, dir_)):
        aux = []
        img = cv.imread(os.path.join(datadir, dir_, img_path))
        # Convert image to RGB so that we can send it to MediaPipe
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # to see if hands exits in img_rgb
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    # print(hand_landmarks.landmark[i])
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    aux.append(x)
                    aux.append(y)
            data.append(aux)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()










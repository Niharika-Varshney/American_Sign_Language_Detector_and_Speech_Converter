import cv2 as cv
import mediapipe as mp
import pickle
import numpy as np
import warnings
import pyttsx3
import concurrent.futures

# Suppress specific UserWarning from google.protobuf.symbol_database
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# Load the model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv.VideoCapture(0)

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Define gesture labels
labels_dict = {
21: "A", 22: "B", 23: "C", 24: "D", 25: "E", 26: "F", 27: "G", 28: "H",
  29: "I", 30: "J", 31: "K", 32: "L", 33: "M", 34: "N", 35: "O", 36: "P",
  37: "Q", 38: "R", 39: "S", 40: "T", 41: "U", 42: "V", 43: "W", 44: "X",
  45: "Y", 46: "Z"
}

# Initialize text-to-speech engine
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def speak(audio):

    engine.say(audio)
    engine.runAndWait()

last_predicted_character = None
executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)

while True:
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if there is an issue with video capture
    H, W, _ = frame.shape

    # Convert frame to RGB
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame,
                                      hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())
            # Collect hand landmark data
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)
        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)
        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        # Ensure data length consistency for the model
        if len(data_aux) == 42:  # Only one hand detected
            data_aux.extend([0] * 42)  # Pad with zeros for the second hand
        elif len(data_aux) > 84:  # More than two hands detected, truncate to two hands
            data_aux = data_aux[:84]

        # Make prediction
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Draw bounding box and predicted text
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=4)
        cv.putText(frame, predicted_character, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv.LINE_AA)

        # Speak the predicted character only if it changes
        if predicted_character != last_predicted_character:
            # Use executor to handle TTS in a separate process
            executor.submit(speak, predicted_character)
            last_predicted_character = predicted_character

    # Display the frame
    cv.imshow('frame', frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break  # Exit loop if 'q' key is pressed

# Release resources
cap.release()
cv.destroyAllWindows()
executor.shutdown()

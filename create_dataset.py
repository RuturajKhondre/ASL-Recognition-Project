import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

print("Starting script...")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

print(f"Looking for data in: {DATA_DIR}")

data = []
labels = []
for dir_ in sorted(os.listdir(DATA_DIR)):
    if not os.path.isdir(os.path.join(DATA_DIR, dir_)):
        continue
    print(f"Processing directory: {dir_}")
    for img_path in sorted(os.listdir(os.path.join(DATA_DIR, dir_))):
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        full_path = os.path.join(DATA_DIR, dir_, img_path)
        print(f"Processing image: {full_path}")
        
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(full_path)
        if img is None:
            print(f"Failed to read image: {full_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)
        else:
            print(f"No hand landmarks detected in {full_path}")

    print(f"Finished processing directory: {dir_}")

print(f"Processed {len(data)} images with detected hand landmarks")

print("Saving data to data.pickle...")
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print("Script completed successfully")

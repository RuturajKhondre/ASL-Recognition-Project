import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import cv2
import mediapipe as mp

DATA_DIR = './data'

data = []
labels = []
classes = [chr(i) for i in range(ord('A'), ord('Y')+1)]  # A to Y

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def process_image(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    data_aux = []
    x_ = []
    y_ = []
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
        return data_aux
    return None

for class_name in classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        print(f"Warning: Directory for class {class_name} not found.")
        continue
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img_data = process_image(img_path)
        if img_data is not None:
            data.append(img_data)
            labels.append(class_name)

data = np.array(data)
labels = np.array(labels)

print(f"Total samples: {len(data)}")
print(f"Unique classes: {np.unique(labels)}")

if len(data) == 0:
    print("No data processed. Please check your data directory and image files.")
    exit()

print("Splitting data...")
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

print("Training model...")
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

print("Evaluating model...")
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

print("Saving model...")
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'labels': classes}, f)

print("Model saved as model.p")

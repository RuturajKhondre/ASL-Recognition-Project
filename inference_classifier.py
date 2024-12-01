import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from collections import deque
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load model and labels dictionary
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
labels_dict = model_dict['labels_dict']

# Define word mappings
WORD_MAPPINGS = {
    'YASH': 'yash',
    'CAT': 'cat',
    'DON': 'don',
    'KING': 'king',
    # Add more word mappings as needed
}

# Modify these constants
LETTER_CAPTURE_DELAY = 5  # Seconds to wait between letter captures
BUFFER_MAX_LENGTH = 4    # Maximum letters to store (e.g., for "YASH")
WORD_CONFIDENCE_THRESHOLD = 5  # Number of consistent frames needed to confirm a letter

# Initialize variables
sign_buffer = deque(maxlen=BUFFER_MAX_LENGTH)
last_capture_time = time.time()  # Initialize with current time
current_letter_counter = {}  # To track letter consistency
last_spoken_time = 0
SPEAK_COOLDOWN = 2
last_captured_letter = None  # Track the last letter we captured

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

def check_word_formation(buffer):
    current_sequence = ''.join(list(buffer))
    for word, pronunciation in WORD_MAPPINGS.items():
        if current_sequence == word:
            return pronunciation
    return None

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        try:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            
            current_time = time.time()
            
            # Count consecutive occurrences of the same letter
            if predicted_character in current_letter_counter:
                current_letter_counter[predicted_character] += 1
            else:
                current_letter_counter = {k: 0 for k in current_letter_counter}  # Reset other counters
                current_letter_counter[predicted_character] = 1

            # Debug information
            print(f"Current letter: {predicted_character}, Count: {current_letter_counter.get(predicted_character, 0)}")
            print(f"Time since last capture: {current_time - last_capture_time:.1f}s")
            
            # Check if we should capture a new letter
            if (current_letter_counter[predicted_character] >= WORD_CONFIDENCE_THRESHOLD and 
                current_time - last_capture_time >= LETTER_CAPTURE_DELAY and 
                predicted_character != last_captured_letter):
                
                print(f"Capturing letter: {predicted_character}")
                
                # Add letter to buffer
                sign_buffer.append(predicted_character)
                last_capture_time = current_time
                last_captured_letter = predicted_character
                
                # Play sound for the captured letter
                try:
                    engine.say(predicted_character)
                    engine.runAndWait()
                except Exception as e:
                    print(f"Speech error: {e}")
                
                # Check if we formed a word
                word = check_word_formation(sign_buffer)
                if word and current_time - last_spoken_time >= SPEAK_COOLDOWN:
                    print(f"Word detected: {word}")
                    try:
                        engine.say(word)
                        engine.runAndWait()
                    except Exception as e:
                        print(f"Speech error: {e}")
                    last_spoken_time = current_time
                    sign_buffer.clear()  # Clear buffer after speaking word
                    last_captured_letter = None  # Reset last captured letter
            
            # Display information on screen
            buffer_text = ''.join(list(sign_buffer))
            time_to_next = max(0, LETTER_CAPTURE_DELAY - (current_time - last_capture_time))
            
            # Make the displays more prominent
            cv2.putText(frame, f"Buffer: {buffer_text}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Next capture in: {time_to_next:.1f}s", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Current: {predicted_character}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display current detected character
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                       cv2.LINE_AA)

        except Exception as e:
            print(f"Prediction error: {e}")

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
engine.stop()
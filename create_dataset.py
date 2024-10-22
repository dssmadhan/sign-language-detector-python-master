import os
import pickle
import mediapipe as mp
import cv2
import warnings

# Suppress the specific warning from google.protobuf about the deprecated method
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Loop through each directory in the data folder
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    
    # Skip if not a directory
    if not os.path.isdir(dir_path):
        continue
    
    # Loop through each image in the directory
    for img_path in os.listdir(dir_path):
        img_full_path = os.path.join(dir_path, img_path)
        
        # Skip non-image files (you can add more extensions as needed)
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Load the image
        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Warning: Unable to load image {img_full_path}")
            continue

        # Convert the image to RGB as required by MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to find hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []  # Reset for each hand

                x_ = []
                y_ = []

                # Extract landmarks
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y

                    x_.append(x)
                    y_.append(y)

                # Normalize the landmarks based on minimum values
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                # Append the data and corresponding label
                data.append(data_aux)
                labels.append(dir_)

# Save the collected data and labels using pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Cleanup MediaPipe resources
hands.close()

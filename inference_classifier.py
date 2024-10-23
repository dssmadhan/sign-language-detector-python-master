import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use default camera index (change if needed)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.5)

# Label mapping for 16 classes (A-P)
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'I', 5: 'J', 6: 'R', 7: 'S',
    8: 'V', 9: 'Y', 10: 'love you', 11: 'Hi', 12: '😀', 13: '♥️', 14: '.', 15: ' '
}

while True:
    try:
        data_aux = []
        x_ = []
        y_ = []

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame was captured successfully
        if not ret or frame is None:
            print("Warning: Failed to capture frame.")
            continue

        # Get frame dimensions
        H, W, _ = frame.shape

        # Convert the frame to RGB as required by MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hand landmarks
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            for hand_landmarks in results.multi_hand_landmarks:
                # Collect x and y coordinates
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                # Normalize data for model input
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            # Calculate bounding box for the hand
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Make a prediction using the model
            prediction = model.predict([np.asarray(data_aux)])

            # Validate predicted index
            if int(prediction[0]) in labels_dict:
                predicted_character = labels_dict[int(prediction[0])]
            else:
                predicted_character = "Unknown"

            # Draw the bounding box and the predicted character
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
        else:
            print("No hand landmarks detected.")

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()

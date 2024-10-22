import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import time

def load_model(model_path):
    """Load the trained model from a pickle file."""
    with open(model_path, 'rb') as file:
        model_dict = pickle.load(file)
    return model_dict['model']

def initialize_mediapipe():
    """Initialize and return MediaPipe Hands."""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
    return hands

def process_frame_for_emoji(frame, hands, model, labels_dict, emoji_dict):
    """Process each video frame and make predictions for emoji recognition."""
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    detected_emoji = None
    gesture_to_clear = 'Z'  # Example gesture to clear the sequence

    if results.multi_hand_landmarks:
        data_aux = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            
            min_x, min_y = min(x_), min(y_)
            data_aux = [(lm.x - min_x, lm.y - min_y) for lm in hand_landmarks.landmark]
            
            # Flatten the list and ensure the length matches the model input
            data_aux = [item for sublist in data_aux for item in sublist]
            if len(data_aux) == 42:  # Adjust based on your model's requirement
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    detected_character = labels_dict[int(prediction[0])]
                    detected_emoji = emoji_dict.get(detected_character, None)
                    
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    
                    # Display the detected emoji
                    if detected_emoji:
                        cv2.putText(frame, detected_emoji, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                    
                    # Check if the detected character is the gesture to clear
                    if detected_character == gesture_to_clear:
                        detected_character = None
                        st.session_state.detected_characters_emoji = []  # Clear the sequence
                        
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    
    return frame, detected_emoji

def run_emoji_recognition():
    st.title('Hand Gesture Recognition for Emojis')

    model = load_model('./model.p')
    hands = initialize_mediapipe()

    # Emoji mapping for gestures
    labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'Z'}
    emoji_dict = {
        'A': '✊',  # Example: Thumbs Up emoji
        'B': '✋',  # Example: Victory Hand emoji
        'C': '👌',  # Example: OK Hand emoji
        'Z': '❌'   # Example: Cross Mark emoji
    }

    # Initialize session state for control
    if 'capturing_emoji' not in st.session_state:
        st.session_state.capturing_emoji = False
    if 'detected_characters_emoji' not in st.session_state:
        st.session_state.detected_characters_emoji = []
    if 'last_detection_time_emoji' not in st.session_state:
        st.session_state.last_detection_time_emoji = 0
    if 'last_display_time_emoji' not in st.session_state:
        st.session_state.last_display_time_emoji = time.time()
    if 'last_detected_character_emoji' not in st.session_state:
        st.session_state.last_detected_character_emoji = None

    start_button = st.button('Start Emoji Recognition')
    stop_button = st.button('Stop Emoji Recognition')
    
    if start_button:
        st.session_state.capturing_emoji = True
        st.session_state.detected_characters_emoji = []
        st.session_state.last_detection_time_emoji = time.time()  # Reset time on start
    
    if stop_button:
        st.session_state.capturing_emoji = False

    if st.session_state.capturing_emoji:
        st.text('Capturing video... Press "Stop" to end.')
        cap = cv2.VideoCapture(0)  # Use default camera index (change if needed)

        if not cap.isOpened():
            st.error("Error: Could not open video source.")
            return

        frame_placeholder = st.empty()
        detection_text_placeholder = st.empty()  # Placeholder for the detected emojis
        
        while st.session_state.capturing_emoji:
            ret, frame = cap.read()

            if not ret or frame is None:
                st.warning("Failed to capture frame.")
                continue

            try:
                frame, detected_emoji = process_frame_for_emoji(frame, hands, model, labels_dict, emoji_dict)
                
                current_time = time.time()
                
                if detected_emoji and detected_emoji != st.session_state.last_detected_character_emoji:
                    if (current_time - st.session_state.last_detection_time_emoji) > 1:  # Detect emoji every 1 second
                        st.session_state.detected_characters_emoji.append(detected_emoji)
                        st.session_state.last_detection_time_emoji = current_time
                
                # Control the display timing of detected emojis
                if (current_time - st.session_state.last_display_time_emoji) > 2:  # Update display every 2 seconds
                    if st.session_state.detected_characters_emoji:
                        detection_text_placeholder.text("Detected Emojis: " + ''.join(st.session_state.detected_characters_emoji))
                        st.session_state.last_display_time_emoji = current_time
                
                # Convert the frame to RGB and display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                frame_placeholder.image(image, channels='RGB', use_column_width=True)
                
                st.session_state.last_detected_character_emoji = detected_emoji
                
            except Exception as e:
                st.error(f"Error during processing: {e}")
                continue

        cap.release()
        cv2.destroyAllWindows()

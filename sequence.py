import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import time
import requests  # Import the requests library

# In-memory user storage (or you can use a database)
users_db = {}

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

def process_frame(frame, hands, model, labels_dict):
    """Process each video frame and make predictions."""
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    detected_character = None

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
                    
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, detected_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    
    return frame, detected_character

def send_to_backend(username, gesture_password, endpoint):
    """Send data to the backend."""
    try:
        response = requests.post(endpoint, json={'username': username, 'password': gesture_password})
        response.raise_for_status()
        st.success(response.json().get('msg', 'Success'))
    except requests.RequestException as e:
        st.error(f"Failed to send data to backend: {e}")
def initialize_session_state():
    """Initialize the session state attributes if they don't exist."""
    if 'capturing' not in st.session_state:
        st.session_state.capturing = False
    if 'detected_characters' not in st.session_state:
        st.session_state.detected_characters = []
    if 'last_detection_time' not in st.session_state:
        st.session_state.last_detection_time = time.time()
    if 'last_detected_character' not in st.session_state:
        st.session_state.last_detected_character = None
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

def sign_up_page(model, hands, labels_dict):
    """Display sign-up page and save the user's gesture password."""
    st.title('Sign Up with Hand Gestures')

    # Initialize session state
    initialize_session_state()

    username = st.text_input('Choose a Username')

    if username in users_db:
        st.warning("Username already exists. Please choose another username.")
        return

    if st.button('Start Camera to Set Gesture Password'):
        st.session_state.capturing = True
        st.session_state.detected_characters = []
        st.session_state.last_detection_time = time.time()  # Reset time on start
        st.session_state.last_detected_character = None  # Initialize last_detected_character

    if st.button('Stop Camera'):
        st.session_state.capturing = False

    if st.session_state.capturing:
        st.text('Capturing gestures... Press "Stop Camera" to end.')
        cap = cv2.VideoCapture(0)  # Use default camera index (change if needed)

        if not cap.isOpened():
            st.error("Error: Could not open video source.")
            return

        frame_placeholder = st.empty()
        detection_text_placeholder = st.empty()  # Placeholder for the detected characters text

        while st.session_state.capturing:
            ret, frame = cap.read()

            if not ret or frame is None:
                st.warning("Failed to capture frame.")
                continue

            try:
                frame, detected_character = process_frame(frame, hands, model, labels_dict)
                
                current_time = time.time()
                
                if detected_character and detected_character != st.session_state.last_detected_character:
                    if (current_time - st.session_state.last_detection_time) > 1:  # Detect character every 1 second
                        st.session_state.detected_characters.append(detected_character)
                        st.session_state.last_detection_time = current_time
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                frame_placeholder.image(image, channels='RGB', use_column_width=True)
                
                detection_text_placeholder.text("Detected Characters: " + ''.join(st.session_state.detected_characters))
                st.session_state.last_detected_character = detected_character

            except Exception as e:
                st.error(f"Error during processing: {e}")
                continue

        cap.release()
        cv2.destroyAllWindows()

    # After stopping camera, display the Sign Up button to store the user and gesture password
    if len(st.session_state.detected_characters) > 0:
        gesture_password = ''.join(st.session_state.detected_characters)

        if st.button('Sign Up'):
            # Send gesture password to backend and save to MongoDB
            users_db[username] = gesture_password
            send_to_backend(username, gesture_password, 'http://localhost:3000/signup')  # Adjust with correct URL
            st.success(f"User {username} registered successfully with gesture password.")
            st.session_state.detected_characters = []

def login_page(model, hands, labels_dict):
    """Display login page and authenticate using gesture sequence as password."""
    st.title('Login with Hand Gestures')

    username = st.text_input('Username')

    if username not in users_db:
        st.warning("Username not found. Please sign up first.")
        return
    
    gesture_password = users_db[username]  # Fetch the stored gesture password for the user

    if st.button('Start Camera for Password'):
        st.session_state.capturing = True
        st.session_state.detected_characters = []
        st.session_state.last_detection_time = time.time()  # Reset time on start
        st.session_state.last_detected_character = None  # Initialize last_detected_character

    if st.button('Stop Camera'):
        st.session_state.capturing = False

    if 'capturing' in st.session_state and st.session_state.capturing:
        st.text('Capturing gestures... Press "Stop Camera" to end.')
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Error: Could not open video source.")
            return

        frame_placeholder = st.empty()
        detection_text_placeholder = st.empty()

        while st.session_state.capturing:
            ret, frame = cap.read()

            if not ret or frame is None:
                st.warning("Failed to capture frame.")
                continue

            try:
                frame, detected_character = process_frame(frame, hands, model, labels_dict)
                
                current_time = time.time()
                
                if detected_character and detected_character != st.session_state.last_detected_character:
                    if (current_time - st.session_state.last_detection_time) > 1:
                        st.session_state.detected_characters.append(detected_character)
                        st.session_state.last_detection_time = current_time
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                frame_placeholder.image(image, channels='RGB', use_column_width=True)
                
                detection_text_placeholder.text("Detected Characters: " + ''.join(st.session_state.detected_characters))
                st.session_state.last_detected_character = detected_character

                if ''.join(st.session_state.detected_characters) == gesture_password:
                    st.session_state.logged_in = True
                    st.success("Logged in successfully!")
                    st.session_state.capturing = False
                    st.session_state.detected_characters = []
                    send_to_backend(username, gesture_password, 'http://localhost:3000/login')  # Update with the correct URL
                    
            except Exception as e:
                st.error(f"Error during processing: {e}")
                continue

        cap.release()
        cv2.destroyAllWindows()

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    model = load_model('./model.p')
    hands = initialize_mediapipe()
    labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'Z'}  # Example labels

    page = st.sidebar.selectbox("Choose Action", ["Login", "Sign Up"])

    if page == "Login":
        login_page(model, hands, labels_dict)
    elif page == "Sign Up":
        sign_up_page(model, hands, labels_dict)

    if st.session_state.logged_in:
        st.title('Hand Gesture Recognition')
        # Rest of the app for hand gesture recognition

if __name__ == "__main__":
    main()

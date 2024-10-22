import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import time
import requests

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Label mapping and corresponding emoji mapping
labels_dict = {0: "hi", 1: 'B', 2: 'C'}
emoji_dict = {0: "👋", 1: '🅱️', 2: '🌜'}  # Add more emoji mappings based on your labels

# Initialize Streamlit session state for storing detected characters and emojis
if 'detected_characters' not in st.session_state:
    st.session_state.detected_characters = []
if 'detected_emojis' not in st.session_state:
    st.session_state.detected_emojis = []
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = time.time()
if 'api_response' not in st.session_state:
    st.session_state.api_response = ""

def process_frame(frame):
    """Process the video frame to detect hand landmarks and predict characters."""
    data_aux = []
    x_ = []
    y_ = []
    
    # Convert the frame to RGB as required by MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)
    predicted_character = None
    predicted_emoji = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
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

        # Make a prediction using the model
        if len(data_aux) == 42:  # Ensure that the input matches the model's input shape
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            predicted_emoji = emoji_dict[int(prediction[0])]
            
    return frame, predicted_character, predicted_emoji

# Streamlit App UI
st.title("Hand Gesture Alphabet and Emoji Recognition")

# Start capturing the video feed from the webcam
run = st.checkbox('Start Camera')

# Display the detected sequence of alphabets and emojis
st.text('Detected Characters and Emojis Sequence:')
detected_characters_placeholder = st.empty()
detected_emojis_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)  # Use default camera index
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open video source.")
    else:
        # Display video feed and process hand gestures
        frame_placeholder = st.empty()

        while run:
            ret, frame = cap.read()

            if not ret:
                st.warning("Failed to capture frame.")
                break

            # Process the frame and predict the character and emoji
            frame, predicted_character, predicted_emoji = process_frame(frame)

            # Display the frame in the Streamlit app
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            frame_placeholder.image(image, use_column_width=True)

            # Check if enough time has passed (e.g., 2 seconds) to add the next character
            current_time = time.time()
            if predicted_character and (current_time - st.session_state.last_detection_time) > 2:
                # Add the character and emoji to the sequence and reset the detection timer
                st.session_state.detected_characters.append(predicted_character)
                st.session_state.detected_emojis.append(predicted_emoji)
                st.session_state.last_detection_time = current_time
            
            # Update the detected characters and emojis display
            detected_characters_placeholder.text(' '.join(st.session_state.detected_characters))
            detected_emojis_placeholder.text(' '.join(st.session_state.detected_emojis))

            # Break the loop on 'q' key press in terminal (since Streamlit does not handle key events like OpenCV)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

# Submit button to send API request
if st.button('Submit Detected Characters and Emojis'):
    api_endpoint = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }

    print("\nDetected characters:", st.session_state.detected_characters)
    print("Detected emojis:", st.session_state.detected_emojis)

    api_payload = {
        "model": "llama3.1",
        "prompt": " ".join(st.session_state.detected_characters) + " " + " ".join(st.session_state.detected_emojis),  # Combining detected characters and emojis
        "options": {
            "temperature": 0
        },
        "stream": False
    }

    print("Sending request to Ollama")
    response = requests.post(api_endpoint, headers=headers, json=api_payload)

    if response.status_code == 200:
        response_data = response.json()
        print(response_data)
        st.session_state.api_response = response_data.get('response', 'No response from Ollama')
        st.success("Response received from Ollama!")
    else:
        print(response.status_code)
        st.error(f"API request failed with status code {response.status_code}")

# Display the Ollama API response
if st.session_state.api_response:
    st.text("Ollama's Response:")
    st.write(st.session_state.api_response)

# Clear the detected characters and emojis list
if st.button('Clear Detected Characters and Emojis'):
    st.session_state.detected_characters = []
    st.session_state.detected_emojis = []
    st.session_state.api_response = ""
    detected_characters_placeholder.text('')
    detected_emojis_placeholder.text('')
    st.success("Detected characters and emojis cleared.")
cv2.destroyAllWindows()

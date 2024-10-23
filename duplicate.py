import streamlit as st
import subprocess
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import time
import requests
import pyttsx3
import threading

# Global variable to hold the subprocesses
processes = {}

# Functions to run specific projects
def run_presentation_project():
    processes['presentation'] = subprocess.Popen(
        ["python3", "/Users/madhandhanasekaran/Documents/PROJ/sign-language-detector-python-master/presentation/main.py"]
    )

def run_mouse_project():
    processes['mouse'] = subprocess.Popen(
        ["python3", "/Users/madhandhanasekaran/Documents/PROJ/sign-language-detector-python-master/mouse/mouse.py"]
    )

def run_volume_project():
    processes['volume'] = subprocess.Popen(
        ["python3", "/Users/madhandhanasekaran/Documents/PROJ/sign-language-detector-python-master/volume/volume.py"]
    )

# Function to stop running projects
def stop_project(project_name):
    if project_name in processes and processes[project_name].poll() is None:
        processes[project_name].terminate()
        st.write(f"{project_name.capitalize()} project stopped.")
    else:
        st.write(f"No running {project_name.capitalize()} project found.")

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Label mapping
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'I', 5: 'J', 6: 'R', 7: 'S',
    8: 'V', 9: 'Y', 10: 'love you', 11: 'Hi', 12: '😀', 13: '♥️', 14: '.', 15: ' '
}

# Initialize Streamlit session state for storing detected characters
if 'detected_characters' not in st.session_state:
    st.session_state.detected_characters = []
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = time.time()
if 'api_response' not in st.session_state:
    st.session_state.api_response = ""

# Initialize a lock for thread safety
speech_lock = threading.Lock()

def speak_text(text):
    """Convert text to speech."""
    def speak():
        with speech_lock:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # Adjust the speed of speech
            engine.say(text)
            engine.runAndWait()

    # Run in a separate thread to avoid blocking the UI
    thread = threading.Thread(target=speak)
    thread.start()

# Function to process the video frame to detect hand landmarks and predict characters
def process_frame(frame):
    data_aux = []
    x_ = []
    y_ = []
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    predicted_character = None
    
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

        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            
    return frame, predicted_character

# Function to submit the concatenated detected characters to the Ollama API
def submit_to_ollama():
    # Concatenate all detected characters into one string
    detected_string = ''.join(st.session_state.detected_characters)
    
    if not detected_string:
        st.warning("No characters detected to send to Ollama.")
        return

    api_endpoint = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}

    api_payload = {
        "model": "llama3.1",
        "prompt": detected_string,  # Send the entire concatenated string as prompt
        "options": {"temperature": 0},
        "stream": False
    }

    response = requests.post(api_endpoint, headers=headers, json=api_payload)

    if response.status_code == 200:
        response_data = response.json()
        st.session_state.api_response = response_data.get('response', 'No response from Ollama')
        st.success("Response received from Ollama!")

        # Only speak if not already speaking
        if not speech_lock.locked():
            speak_text(st.session_state.api_response)

        # Display Ollama API response in the Streamlit interface
        st.text("Ollama's Response:")
        st.write(st.session_state.api_response)

    else:
        st.error(f"API request failed with status code {response.status_code}")

# Main function for the Streamlit appdef main():
def main():
    st.title("Sign Language Detector and Gesture Recognition")
    
    # Hand Gesture Recognition
    st.header("Hand Gesture Recognition")
    
    # Start capturing the video feed from the webcam
    run = st.checkbox('Start Camera')

    # Display the detected sequence of alphabets
    st.text('Detected Characters Sequence:')
    detected_characters_placeholder = st.empty()

    # Clear the detected characters list button should always be visible
    if st.button('Clear Detected Characters'):
        st.session_state.detected_characters = []
        st.session_state.api_response = ""
        detected_characters_placeholder.text('')
        st.success("Detected characters cleared.")
    
    if run:
        cap = cv2.VideoCapture(0)  # Use default camera index

        if not cap.isOpened():
            st.error("Error: Could not open video source.")
        else:
            frame_placeholder = st.empty()

            while run:
                ret, frame = cap.read()

                if not ret:
                    st.warning("Failed to capture frame.")
                    break

                # Process the frame and predict the character
                frame, predicted_character = process_frame(frame)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                frame_placeholder.image(image, use_column_width=True)

                current_time = time.time()
                if predicted_character and (current_time - st.session_state.last_detection_time) > 2:
                    st.session_state.detected_characters.append(predicted_character)
                    st.session_state.last_detection_time = current_time

                    # If the detected character is ".", automatically submit the detected characters
                    if predicted_character == ".":
                        st.write("Detected '.' gesture. Submitting detected characters...")
                        submit_to_ollama()

                detected_characters_placeholder.text(' '.join(st.session_state.detected_characters))

            cap.release()

    cv2.destroyAllWindows()

    # Project Management Section
    st.header("Project Management")

    # Presentation Project
    col1, col2 = st.columns(2)
    with col1:
        button_style = """
        <style>
        .stButton > button {
            background-color: #88217a;
            color: white;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 12px;
            transition: background-color 0.3s, color 0.3s;
        }
        .stButton > button:hover {
            background-color: white;
            color: #88217a;
        }
        </style>
        """
        st.markdown(button_style, unsafe_allow_html=True)
        
        if st.button("Run Presentation Project"):
            st.write("Running Presentation Project...")
            run_presentation_project()

    # Mouse Control Project
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Run Mouse Control Project"):
            st.write("Running Mouse Control Project...")
            run_mouse_project()

    # Volume Control Project
    col5, col6 = st.columns(2)
    with col5:
        if st.button("Run Volume Control Project"):
            st.write("Running Volume Control Project...")
            run_volume_project()

if __name__ == "__main__":
    main()

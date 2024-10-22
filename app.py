from flask import Flask, render_template, Response, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import requests

app = Flask(__name__)

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Label mapping
labels_dict = {0: 'A', 1: 'B', 2: 'C'}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

detected_characters = []
last_detection_time = time.time()  # To store the time of the last detection

def process_frame(frame):
    """Process a single frame to detect hand landmarks and predict gesture."""
    global detected_characters, last_detection_time
    data_aux = []
    x_ = []
    y_ = []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_character = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))

        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

        current_time = time.time()
        if current_time - last_detection_time >= 2 and predicted_character:
            detected_characters.append(predicted_character)
            last_detection_time = current_time

    return frame

def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detected_characters')
def get_detected_characters():
    global detected_characters
    return jsonify(detected_characters=detected_characters)

@app.route('/clear_characters')
def clear_characters():
    global detected_characters
    detected_characters = []
    return jsonify({"message": "Detected characters cleared."})

@app.route('/send_to_ollama')
def send_to_ollama():
    global detected_characters
    if detected_characters:
        api_endpoint = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        api_payload = {
            "model": "llama3.1",
            "prompt": " ".join(detected_characters),
            "options": {"temperature": 0},
            "stream": False
        }
        try:
            response = requests.post(api_endpoint, headers=headers, json=api_payload)
            if response.status_code == 200:
                response_data = response.json()
                return jsonify({"message": "Success", "response": response_data.get('response', 'No response from Ollama')})
            else:
                return jsonify({"message": f"Failed with status code {response.status_code}"}), response.status_code
        except requests.exceptions.RequestException as e:
            return jsonify({"message": f"Request error: {str(e)}"}), 500
    else:
        return jsonify({"message": "No detected characters to send."}), 400

if __name__ == '__main__':
    app.run(debug=True)

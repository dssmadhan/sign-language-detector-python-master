import os
import cv2

# Directory to store the dataset
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 16  # Total number of classes for data collection
dataset_size = 150  # Number of images per class

# Attempt to use the camera at index 2
cap = cv2.VideoCapture(2)

# Check if the camera opened successfully, otherwise fallback to the default camera
if not cap.isOpened():
    print("Error: Could not open camera at index 2. Trying default camera...")
    cap = cv2.VideoCapture(0)  # Try the default camera
    if not cap.isOpened():
        print("Error: Could not open any video source. Exiting...")
        exit()

# Iterate over each class for data collection
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Wait for the user to press 'Q' to start capturing for each class
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Couldn't capture frame. Trying again...")
            continue

        # Display instructions for the user
        cv2.putText(frame, 'Press "Q" to start capturing for class ' + str(j), 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Wait for the user to press 'Q' to proceed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Start capturing images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Couldn't capture frame. Skipping...")
            continue

        # Display the live video feed during capture
        cv2.putText(frame, f'Capturing {counter+1}/{dataset_size} images', 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Save the captured frame as an image
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

        # Allow exiting the capture loop early if the user presses 'Q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture interrupted. Moving to the next class...")
            break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 10
dataset_size = 150

# Attempt to use the camera at index 2
cap = cv2.VideoCapture(2)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera at index 2. Trying default camera...")
    cap = cv2.VideoCapture(0)  # Try the default camera
    if not cap.isOpened():
        print("Error: Could not open any video source.")
        exit()

# Iterate over the classes
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Wait for the user to press 'Q' to start capturing
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Couldn't capture frame.")
            continue

        cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Wait for the user to press 'Q' to proceed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Capture images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Couldn't capture frame. Skipping...")
            continue

        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)  # Save captured frame as an image
        counter += 1

        # Introduce a small delay for capturing images
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

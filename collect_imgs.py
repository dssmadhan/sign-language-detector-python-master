import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

cap = cv2.VideoCapture(2)  # Attempt to use camera at index 2

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video source. Trying default camera...")
    cap = cv2.VideoCapture(0)  # Try the default camera
    if not cap.isOpened():
        print("Error: Could not open any video source.")
        exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Wait for user to start capturing
    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Warning: Couldn't capture frame.")
            continue

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Capture images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Warning: Couldn't capture frame. Skipping...")
            continue

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()

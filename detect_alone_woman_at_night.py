import cv2
import numpy as np
from datetime import datetime

# Paths to your model files
yolo_weights_path = 'yolov3.weights'    # Change this to your actual path
yolo_config_path = 'yolov3.cfg'         # Change this to your actual path
coco_names_path = 'coco.names'        # Change this to your actual path
gender_prototxt = 'gender_deploy.prototxt'  # Path to your deploy prototxt
gender_model = 'gender_net.caffemodel'     # Path to your gender caffemodel
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # Haar cascade for face detection

# Load YOLO model
net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)

# Get the output layer names of YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load Gender Classification Model
gender_net = cv2.dnn.readNetFromCaffe(gender_prototxt, gender_model)
gender_list = ['Male', 'Female']

# Load Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Function to detect people and classify gender at night
def detect_person_and_classify_gender(frame):
    height, width = frame.shape[:2]

    # Prepare the image for the YOLO model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Loop through each output layer
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections and only keep persons
            if confidence > 0.5 and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to suppress overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    now = datetime.now()
    is_night = now.hour >= 13 or now.hour < 6  # Assuming it's night between 8 PM and 6 AM

    # Count the number of people detected
    num_people = len(indexes.flatten()) if len(indexes) > 0 else 0

    # Proceed only if one person is detected and it's night-time
    if is_night and num_people == 1:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]

            # Detect face within the person's bounding box
            person_roi = frame[y:y + h, x:x + w]
            gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Process only if at least one face is detected
            for (fx, fy, fw, fh) in faces:
                face = person_roi[fy:fy + fh, fx:fx + fw]

                # Resize the face region for the gender classification model
                face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104.0, 177.0, 123.0), swapRB=False)
                gender_net.setInput(face_blob)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]

                # Only display the bounding box and label if the detected person is Female
                if gender == "Female":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for Female
                    label = f"{gender}: {confidences[i]:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Logging detected information
                    print(f"Detected {gender} with confidence {confidences[i]:.2f} at night.")

    return frame

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect persons and classify gender in the current frame
    detected_frame = detect_person_and_classify_gender(frame)

    # Display the frame with detections and classifications
    cv2.imshow('Alone Female Detection at Night', detected_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
import cv2
import os
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import time

# Load the trained YOLO model
model_path = "runs/detect/train3/weights/best.pt"  # Update if needed
model = YOLO(model_path)

# Get class names from the model automatically
class_names = model.names

# Open webcam
cap = cv2.VideoCapture(0)  # 0 for default camera, change if needed

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

print("Starting real-time helmet violation detection...")

# For tracking violations
violation_count = 0
last_violation_time = 0
violation_cooldown = 5  # seconds between logging the same violation

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO inference
    results = model(frame, conf=0.5)  # Adjust confidence threshold as needed

    # Extract detections
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes else []

    # Initialize flags
    rider_detected = False
    helmet_detected = False
    plate_regions = []

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        cls = int(cls)
        label = class_names[cls]

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check for rider and helmet
        if label == "rider":
            rider_detected = True
        elif label == "with helmet":
            helmet_detected = True
        elif label == "number plate":
            # Store plate region for OCR
            plate_regions.append((int(x1), int(y1), int(x2), int(y2)))

    # Check for violation: rider without helmet
    violation = rider_detected and not helmet_detected

    if violation:
        current_time = time.time()
        if current_time - last_violation_time > violation_cooldown:
            violation_count += 1
            last_violation_time = current_time
            print(f"Violation detected! Total violations: {violation_count}")

            # Draw violation alert
            cv2.putText(frame, "VIOLATION: No Helmet!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Extract number plate text for violations
            for plate_region in plate_regions:
                x1, y1, x2, y2 = plate_region
                plate_img = frame[y1:y2, x1:x2]
                if plate_img.size > 0:
                    # Use PaddleOCR for text recognition
                    result = ocr.ocr(plate_img, cls=True)
                    if result and result[0]:
                        plate_text = ""
                        confidence = 0
                        for line in result[0]:
                            text, conf = line[1]
                            plate_text += text + " "
                            confidence = max(confidence, conf)
                        plate_text = plate_text.strip()
                        if plate_text:
                            print(f"Plate text: {plate_text} (confidence: {confidence:.2f})")
                            cv2.putText(frame, f"Plate: {plate_text}", (x1, y1-30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display frame
    cv2.imshow("Helmet Violation Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(f"Detection stopped. Total violations detected: {violation_count}")

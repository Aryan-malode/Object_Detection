# Step 1: Install the necessary libraries


# Step 2: Import the required libraries
import cv2
import torch
from ultralytics import YOLO

# Step 3: Load the YOLOv8 pre-trained model
model = YOLO('yolov8m.pt')  # You can use 'yolov8m.pt' for more accuracy (medium model)

# Step 4: Capture video from the webcam
cap = cv2.VideoCapture(0)  # '0' is the default ID for webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Step 5: Define the codec and output for saving the video (Optional)
output_path = 'output_live_video1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
out = None

# Step 6: Process the live video feed frame by frame
while True:
    ret, frame = cap.read()  # Capture a frame from the webcam

    if not ret:
        print("Failed to grab frame")
        break

    # Perform object detection on the frame
    results = model(frame)

    # Annotate the frame with bounding boxes and labels
    annotated_frame = results[0].plot()

    # Display the frame with detections
    cv2.imshow('YOLOv8 Live Object Detection', annotated_frame)

    # Optional: Save the output video (you can remove this if you don't want to save the video)
    if out is None:
        height, width, _ = frame.shape
        out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    out.write(annotated_frame)  # Write the annotated frame to the output video

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 7: Release the webcam and close all OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Object detection finished. Saved output video to {output_path}")

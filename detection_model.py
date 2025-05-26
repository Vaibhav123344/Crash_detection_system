import cv2
import os
import time
import threading
import winsound  # Windows-only
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('best(2).pt')

# Directory to save detected frames
save_dir = 'detected_accidents'
os.makedirs(save_dir, exist_ok=True)

# Class IDs that should trigger the alarm and image saving
trigger_classes = [0, 1]  # Update according to your data.yaml

# Beep sound for 5 seconds
def play_beep():
    duration = 500  # ms
    frequency = 1000  # Hz
    end_time = time.time() + 5
    while time.time() < end_time:
        winsound.Beep(frequency, duration)

# Start webcam
cap = cv2.VideoCapture(0)

saving = False
save_start_time = 0
save_count = 0
beep_triggered = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model.predict(source=frame, conf=0.5, save=False, verbose=False)
    detections = results[0].boxes

    # Check if any detection is in trigger classes
    if detections is not None and len(detections) > 0:
        trigger_detected = any(int(cls) in trigger_classes for cls in detections.cls)

        if trigger_detected:
            if not saving:
                saving = True
                save_start_time = time.time()
                save_count = 0
                beep_triggered = False
                print("[INFO] Accident condition detected. Saving frames...")

            # Beep once in a background thread
            if not beep_triggered:
                threading.Thread(target=play_beep, daemon=True).start()
                beep_triggered = True

            # Save up to 10 images over 3 seconds
            if time.time() - save_start_time <= 3 and save_count < 10:
                save_path = os.path.join(save_dir, f"accident_{int(time.time()*1000)}.jpg")
                cv2.imwrite(save_path, frame)
                save_count += 1

            if save_count >= 10 or time.time() - save_start_time > 3:
                saving = False
                print("[INFO] Finished saving 10 frames.")
        else:
            saving = False
            beep_triggered = False
    else:
        saving = False
        beep_triggered = False

    # Show frame
    annotated_frame = results[0].plot()
    cv2.imshow("Real-Time Accident Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



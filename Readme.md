# 🚗 Accident Detection Model (YOLOv8)

This repository contains a trained YOLOv8 model for real-time accident detection. The model was trained to detect various types of accidents including **moderate**, **severe**, and **fire-related incidents**.

##  Files

- `best.pt`: Trained YOLOv8 model weights.
- `README.md`: Project description and usage instructions.

##  Model Info

- Framework: [Ultralytics YOLOv8](https://docs.ultralytics.com)
- Dataset: Custom annotated images with bounding boxes for different accident severities
- Classes Detected:
  - `0`: Moderate
  - `1`: Fire
  - `2`: Severe

##  Inference Example (Python)

```python
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Run prediction on an image
results = model.predict(source='test.jpg', show=True)

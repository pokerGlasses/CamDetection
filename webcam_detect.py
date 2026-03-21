import sys
import os
from inference import get_model
import supervision as sv
import cv2

if len(sys.argv) != 2:
    print("Usage: python webcam_detect.py <image.jpg>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.isfile(image_path):
    print(f"File not found: {image_path}")
    sys.exit(1)

frame = cv2.imread(image_path)
if frame is None:
    print(f"Could not read image: {image_path}")
    sys.exit(1)

api_key = "***REMOVED***"
model = get_model(model_id="playing-cards-ow27d/4", api_key=api_key)

results = model.infer(frame)[0]
detections = sv.Detections.from_inference(results)

labels = [results.predictions[i].class_name for i in range(len(detections))]

print("Detected cards:")
for card in labels:
    print(f"  - {card}")

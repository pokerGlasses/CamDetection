import os
from inference import get_model
import supervision as sv
import cv2

# open the default webcam (device 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to open webcam")

# load a pre-trained model once
api_key = "***REMOVED***"
model = get_model(model_id="playing-cards-ow27d/4", api_key=api_key)

# create supervision annotators outside the loop
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# process frames until the user quits
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run inference on the current frame
    results = model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)

    # annotate the frame
    annotated = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections)

    # show the annotated frame in a window
    cv2.imshow("webcam detection", annotated)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import os

thresh = 0.5

# Load class names
classFile = 'coco.names'
if not os.path.exists(classFile):
    print(f"Error: File '{classFile}' not found.")
    exit()

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model files
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'

if not os.path.exists(configPath):
    print(f"Error: File '{configPath}' not found.")
    exit()
if not os.path.exists(weightPath):
    print(f"Error: File '{weightPath}' not found.")
    exit()

# Initialize the detection model
net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to read frame from webcam.")
        break

    classIds, confs, bbox = net.detect(img, confThreshold=thresh)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if 0 < classId <= len(classNames):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0]+10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)) + '%', (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

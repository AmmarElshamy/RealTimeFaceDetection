# import required packages
import numpy as np
import cv2
import argparse
import imutils
from imutils.video import VideoStream
import time

# Construct the argument parse
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-m", "--model", required=True, help="path to Cafe model architecture")
argument_parser.add_argument("-p", "--prototxt", required=True, help="path to Cafe 'weights' protoxt file")
argument_parser.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter"
                                                                                       "weak detections")
args = vars(argument_parser.parse_args())


# load our model
print("[INFO] loading model...")
model = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


# loop over the frames of the video stream
while True:
    # grab the frame and resize
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    # construct an input blob for the frame
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and get detections
    print("[INFO] computing object detections...")
    model.setInput(blob)
    detections = model.forward()

    # loop over the detections of shape [1, 1, n, 7]
    for i in range(0, detections.shape[2]):
        # get the confidence of the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence < args["confidence"]:
            continue

        # compute (x, y) coordinates of the bounding box
        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        (startX, startY, endX, endY) = box.astype("int")

        # draw bounding box of the face with its probability
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        text = f"{confidence * 100: .2f}%"
        y = startY - 10 if startY - 10 > 0 else startY + 10
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # show output frame
    cv2.imshow("Face Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # if 'q' key is pressed, quit
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

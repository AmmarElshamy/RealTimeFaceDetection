# import required packages
import numpy as np
import cv2
import argparse

# Construct the argument parse
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-i", "--image", required=True, help="path to input image")
argument_parser.add_argument("-m", "--model", required=True, help="path to Cafe model architecture")
argument_parser.add_argument("-p", "--prototxt", required=True, help="path to Cafe 'weights' protoxt file")
argument_parser.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter"
                                                                                       "weak detections")
args = vars(argument_parser.parse_args())


# load our model
print("[INFO] loading model...")
model = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


# load the input image and costruct an input blob for the image
image = cv2.imread(args["image"])
(height, width) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))


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
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    text = f"{confidence * 100: .2f}%"
    y = startY - 10 if startY - 10 > 0 else startY + 10
    cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


# show output image
cv2.imshow("Output", image)
cv2.waitKey(0)

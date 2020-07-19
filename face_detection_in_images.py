## using opencv hidden face detector

import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser() # creating an argument parser object
ap.add_argument("-i", "--image", required = True, help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, help="path to prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to caffe weights")
ap.add_argument("-c", "--confidence", type = float, default=0.5, help="minimum probability to filter weak predictions")
args = vars(ap.parse_args())

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]
    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > args["confidence"]:
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
 
        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(img, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(img, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
# show the output image
cv2.imshow("Output", img)
cv2.waitKey(0)
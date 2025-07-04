# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcasscades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box
# 3. Flatten the largest face image and save in a numpy array
# 4. Repeat the above for multiple peple to generate training data

import cv2
import numpy as np


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
    ret, frame = cap.read()

    if ret == False:
        continue
    
    # need to convert to grayscale for face detection

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # remember this
    cascade_coords = face_cascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5 ) 
    for (x, y, w, h) in cascade_coords:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Detection',frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcasscades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box
# 3. Flatten the largest face image and save in a numpy array
# 4. Repeat the above for with or without specs for multiple people

import cv2
import numpy as np

#Initialise the camera
file_name  = input('Are you wearing specs?')
cap = cv2.VideoCapture(0)

#We will be using haarcascade to detect faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
skip = 0
face_data  = []
dataset_path = './data/'
    

while True:

    ret, frame = cap.read()

    #ret is a bool variable that depicts whether the frame is captured correctly or not
    if ret == False:
        continue
    

    #We need to store images as grey scale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    #This owuld give us the list of the face coordinates (x,y,w, h)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors=5)
    

    
    faces = sorted(faces, key = lambda f: f[2]*f[3], reverse = True )
    print(faces)

    # Pick the last face
    for (x,y,w,h) in faces[0:]:
            cv2.rectangle(frame, (x,y), (x+w, y+h),(0, 255, 255), 2  )
            #Extract: Region of interest
            offset = 10 #padding
            face_section = gray[y-offset: y+h+offset, x-offset: x+w+offset ]
            face_section = cv2.resize(face_section, (100, 100))

            skip = skip + 1
            print(skip)
            if skip%10 == 0:
                face_data.append(face_section)
                print(len(face_data))
    
    #Make sure cv2.imshow comes after drawing rectangles as by the time we draw the rectangles the frame is already displayed
    cv2.imshow('Muaz',frame)
        
    
    
    # We need to sort faces based on the largest area
    # we will be storing only the sorted faces


    #store every 10th face
    

    



    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
# convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1 ))
print(face_data.shape)

# Save this data into the file system
np.save(dataset_path + file_name + '.npy', face_data) # function to save numpy array

cap.release()
cv2.destroyAllWindows()


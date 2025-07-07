# Recognise faces some classififcation algorithm - like logistic, knn, svm etc

# 1. Load the training data (numpy array of all thje persons)
        # x-values are stored in numpy arrays
        # y-values we need to assign for each person

# 2. Read a video stream using opencv
# 3. Extract faces out of it (for testing it)
# 4. use knn to find the prediction of face(int)
# 5. map the predicted id to name of the user
# 6. Display the predictions on the screen - bounding box and name

import numpy as np
import cv2
import os


#KNN code

def distance(v1, v2):
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k = 5):
   
    # Get the vector and label
    dist = []

    for i in range(train.shape[0]):
        # Get the vector and the label
        ix = train[i,:-1]
        iy = train[i, -1]

        #Computing the distance from the test point
        d = distance(test, ix)
        dist.append([d, iy])

    # Sort based on distance and get top k
    dk = sorted(dist, key = lambda x: x[0])[:k]
    
    # Converting to numpy array
    labels = np.array(dk)[:, -1]

    # Get frequencies of each label
    output =  np.unique(labels, return_counts = True)

    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]


dataset_path = './data/'
face_data = [] #training data
label = []

class_id  = 0
names = {} #Mapping between id-name

#data preparation

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        #create labels for the class

        target = class_id*np.ones(data_item.shape[0])
        class_id += 1
        label.append(target)

face_dataset = np.concatenate(face_data, axis = 0)
face_labels = np.concatenate(label, axis = 0).reshape((-1, 1)) # converting to column

#combining x labels and y labels as the knn takes that as input

traindataset = np.concatenate((face_dataset, face_labels), axis = 1)
print(traindataset.shape)

# Testing

#Init Camera
cap = cv2.VideoCapture(0)

#Face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0

while True:
    ret, frame = cap.read()

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for face in faces:
        x, y, w, h = face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Get the face ROI
        offset = 10
        face_section = gray[y - offset:y+h+offset, x - offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        #predicted label
        out = knn(traindataset, face_section.flatten())

        pred_name = names[int(out)]
        #display on the screen and rectangle around it
        cv2.putText(frame, pred_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)
    cv2.imshow('Am I wearing specs', frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




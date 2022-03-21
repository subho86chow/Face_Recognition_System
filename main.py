#importing all modules....

from pickle import NONE
import face_recognition as fr
import cv2
import numpy as np
import os

# making list of training models

path = 'train'
images = []
classname = []
mylist = os.listdir(path)

# image text

for cls in mylist:
    currentimg = cv2.imread(f'{path}/{cls}')
    images.append(currentimg)
    classname.append(os.path.splitext(cls)[0])


# finding encodings of the training models

def findencodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodedlist = findencodings(images)

# print(len(encodedlist))

cap = cv2.VideoCapture(0)

# decreasing image quality and detecting face locations
while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    curframeface = fr.face_locations(imgS)
    encodecurframe = fr.face_encodings(imgS,curframeface)

    for encodeface,faceloc in zip(encodecurframe,curframeface):
        matches = fr.compare_faces(encodedlist,encodeface)
        facedistance = fr.face_distance(encodedlist,encodeface)
        # print(facedistance)
        matchindex = np.argmin(facedistance)

# comparing cruuent image with trained models
        if matches[matchindex]:
            name = classname[matchindex].upper()
            # print(name)
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+16,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)


    cv2.imshow('output',img)
    cv2.waitKey(1)

#importing open cv 
from tkinter import font
import cv2
from random import randrange as r

from numpy import size

#dataset load
trainedDataset=cv2.CascadeClassifier('face.xml')

# Starting webcam
webcam=cv2.VideoCapture(0)
while True:

  success,img=webcam.read()

 

  #Converting to black and white(grayscale)
  #BGR-BLUE-GREEN-RED
  grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  #detecting faces in the image
  face_coordinates=trainedDataset.detectMultiScale(grayimg)
  
  for  x,y,w,h in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(r(0,255),r(0,255),r(0,255)),2)
    cv2.putText(img, 'FACE DETECTED', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
    0.9, (0,0,255), 2)
   
  #Displaying the image 
  cv2.imshow('Window',img)
  key=cv2.waitKey(1)
  if(key==27):
      break
webcam.release()
print('END OF PROGRAM')
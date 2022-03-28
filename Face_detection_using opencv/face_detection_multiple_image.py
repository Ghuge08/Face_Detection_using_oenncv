#importing open cv 
import cv2
from random import randrange as r

#dataset load
trainedDataset=cv2.CascadeClassifier('face.xml')

#choose a image
img=cv2.imread('image3.jpg')

#Converting to black and white(grayscale)
#BGR-BLUE-GREEN-RED
grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detecting faces in the image
face_coordinates=trainedDataset.detectMultiScale(grayimg)
#print(face_coordinates) #[[ 80  61 216 216]]  [x,y,width,height]

for  x,y,w,h in face_coordinates:
   cv2.rectangle(img,(x,y),(x+w,y+h),(r(0,255),r(0,255),r(0,255)),2)
   
#Displaying the image 
cv2.imshow('Window',img)
cv2.waitKey()
print('END OF PROGRAM')


import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('haar_face_recognition.xml')

people = ['Jonny Depp', 'Zayn Malik', 'Zendaya']

face_features =np.load('face_features.npy',allow_pickle=True)
face_labeles = np.load('face_labels.npy',allow_pickle=True)


face_recognization = cv.face.LBPHFaceRecognizer_create()
face_recognization.read("face_recognition_model.yml")

img =cv.imread(r'test_photos\jonny_depp.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)


#detect face

face_rect = haar_cascade.detectMultiScale(gray,1.1,4)

for(x,y,w,h) in face_rect:
    face_roi = gray[x:x+w,y:y+h]
    
    label,confidence = face_recognization.predict(face_roi)
    print(f'label = {people[label]} with confedence of {confidence}')
    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255), thickness=2)

cv.imshow("detected",img)
cv.waitKey(0)
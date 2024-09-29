import cv2 as cv
import numpy as np
import os


haar_cascade = cv.CascadeClassifier('haar_face_recognition.xml')
people = ['Jonny Depp', 'Zayn Malik', 'Zendaya']
DIR = r'C:\face recognition\faces_to_train'

features = []
labels = []


def create_train():
    for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)
        
        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            Face_rect = haar_cascade.detectMultiScale(gray,5,4)

            for (x,y,w,h) in Face_rect:
                face_roi = gray[y:y+h,x:x+w]
                features.append(face_roi)
                labels.append(label)

                
create_train()


features = np.array(features,dtype='object')
labels = np.array(labels)

face_recognization = cv.face.LBPHFaceRecognizer_create()
face_recognization.train(features,labels)
face_recognization.save('face_recognition_model.yml')

np.save('face_features.npy',features)
np.save('face_labels',labels)
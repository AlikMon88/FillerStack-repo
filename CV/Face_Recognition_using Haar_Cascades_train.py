#Training_oF_model -----
import os 
import cv2 as cv
import numpy as np

path='cb_face/train/'
people = os.listdir(path)
feature=[]
label=[]
haar_cascade=cv.CascadeClassifier('har_cascade.xml')
def create_train():    
    for cb in people:
        cb_nxt = os.path.join(path,cb)
        labels=people.index(cb)
        for img in os.listdir(cb_nxt):
            img_path=os.path.join(cb_nxt,img)
            img=cv.imread(img_path)
            gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

            faces_rect=haar_cascade.detectMultiScale(gray,1.1,4)

            for (x,y,w,h) in faces_rect:
                img_data=gray[y:y+h,x:x+w]
                feature.append(img_data)
                label.append(labels)
create_train()
feature=np.array(feature)
label=np.array(label)
recgn = cv.face.LBPHFaceRecognizer_create()
recgn.train(feature,label)

recgn.save('face_recog.yml') #Saving_the_trained_model

print('Training Done!!........')


np.save('label.npy',label)
np.save('feature.npy',feature)



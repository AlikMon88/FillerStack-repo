#Testing_on_Videos
import os
path= r'F:\Train_face'
path_v=r'F:\YT videos\elon_musk.mp4'
import cv2 as cv
people=os.listdir(path)
face_r = cv.face.LBPHFaceRecognizer_create()
face_r.read('face_recog.yml')
haar_cascade_face=cv.CascadeClassifier('har_cascade.xml')
capture=cv.VideoCapture(path_v)
while True:
    isTrue, img = capture.read() #img=frame
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face_rect_face=haar_cascade_face.detectMultiScale(gray,1.2,6)
    print(f'{len(face_rect_face)} face/s found !!')
    for (x,y,w,h) in face_rect_face:
        cv.rectangle(img,(x,y),(x+w,y+h),[0,255,0],thickness=3)
        cv.rectangle(img,(x,y),(x+150,y-50),[0,255,0], -1)
        
        cv.putText(img,str(len(face_rect_face))+'Face_detcted',(30,30),cv.FONT_HERSHEY_COMPLEX,1.0,[0,255,0])
        #image_data=gray[y:(y+h),x:x+w]
        
        image_roi=gray[y:y+h,x:x+w]
        label, confidence = face_r.predict(image_roi)
        cv.putText(img,str(people[label]),(x,y-20),cv.FONT_HERSHEY_COMPLEX,0.8,[0,0,0])
        #cv.putText(img,people[label],(30,90),cv.FONT_HERSHEY_COMPLEX,1.0,[0,255,0])
        print(f'{people[label]} with confidence {confidence}')
        
    cv.imshow('Video', img)
    if cv.waitKey(10) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

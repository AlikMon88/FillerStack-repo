#Face_Detection && Eye_Detection ---
path= r'elon_musk.mp4'
import cv2 as cv
haar_cascade_eye=cv.CascadeClassifier('haar_eye_2.xml')
haar_cascade_face=cv.CascadeClassifier('har_cascade.xml')
capture=cv.VideoCapture(path)
while True:
    isTrue, img = capture.read() #img=frame
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face_rect_face=haar_cascade_face.detectMultiScale(gray,1.2,6)
    print(f'{len(face_rect_face)} face/s found !!')
    for (x,y,w,h) in face_rect_face:
        cv.rectangle(img,(x,y),(x+w,y+h),[0,255,0],thickness=3)
        cv.putText(img,str(len(face_rect_face))+'Face_detcted',(30,30),cv.FONT_HERSHEY_COMPLEX,1.0,[0,255,0])
        image_data=gray[y:y+h,x:x+w]
        x2=x
        y2=y
    
    
        face_rect_eye = haar_cascade_eye.detectMultiScale(image_data,1.6,4)
        print(f'{len(face_rect_eye)} eye/s found !!')
        for (x,y,w,h) in face_rect_eye:
            cv.rectangle(img,(x+x2,y+y2),((x+x2)+w,(y+y2)+h),[0,255,0],thickness=3)
            cv.putText(img,str(len(face_rect_eye))+'Eye_detcted',(30,60),cv.FONT_HERSHEY_COMPLEX,1.0,[0,255,0])
            
    
    
    cv.imshow('Final', img)
    if cv.waitKey(10) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

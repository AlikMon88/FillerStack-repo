#Testing_the_trained_model -----
path_t=r'cb_face/httpcsvkmeuaeccjpg.jpg'
img=cv.imread(path_t)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

people=os.listdir(path)

face_r = cv.face.LBPHFaceRecognizer_create()
face_r.read('face_recog.yml')

haar_cascade=cv.CascadeClassifier('har_cascade.xml')
face_rect=haar_cascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in face_rect:
    x_test=gray[y:y+h,x:x+w]#x_test -- passes the facial data
    cv.rectangle(img,(x,y), (x+w,y+h), [0,255,0], thickness=3)
    label, confidence = face_r.predict(x_test)
    cv.putText(img, people[label], (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, [0,255,0])
    print(f'{people[label]} with confidence {confidence}')

cv.imshow('Result_image',img)    
    
cv.waitKey(0)
cv.destroyAllWindows()

import cv2 as cv
import keras
import os
import numpy as np
model = keras.models.load_model('obj_det.pb')
path=r"F:\YT videos\road_traffic.mp4"
with open('coco_names.pbtxt', 'r') as f:
    class_names=f.read()
    class_names=class_names.strip('\n').split('\n')
f.close()
config_path=r"C:\Users\User\Desktop\Jupy2\ssd_mobilenet_v3_coco.pbtxt"
weight_path=r"C:\Users\User\Desktop\Jupy2\frozen_inference_graph.pb"
net = cv.dnn_DetectionModel(weight_path,config_path)
net.setInputSize(328,328)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
cap=cv.VideoCapture(path)
while True:
    isTrue, img = cap.read()
    img=cv.resize(img,(600,600))
    #img=np.flip(img,1)
    classIds, confs, bbox = net.detect(img, confThreshold=0.55)
    #print(bbox)
    
    if type(bbox)==tuple:
        continue
    
    if  bbox.size!=0:
        box=bbox.tolist()
        for ((x,y,w,h),i) in zip(box,classIds):
            if h*w >= 9000:
                continue
            cv.rectangle(img,(x,y),(x+w,y+h), [0,255,0], 2)
            for j in i:
                cv.rectangle(img,(x,y),(x+w,y-30),[0,255,0], -1)
                cv.putText(img,str(class_names[j-1]),(x,y-20),cv.FONT_HERSHEY_COMPLEX,0.5,[0,0,0])
                print(f'{class_names[j-1]} with confidence {confs}')
        cv.imshow('Detected_img',img)
    if cv.waitKey(10) & 0xFF==ord('d'):
        break
cap.release()
cv.destroyAllWindows()

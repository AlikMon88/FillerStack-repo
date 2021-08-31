# Invisibility Cloak
import numpy as np
import cv2 

background=0
count=0

def rescaleframe(frame, scale=0.75):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimension = (width,height)
    return cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)        
        
cap=cv2.VideoCapture('Video.mp4')
for i in range(60):
    ret,background=cap.read()
    background=rescaleframe(background)

while (cap.isOpened()):
    ret,img=cap.read()
    if ret == False:
        break
    img=rescaleframe(img)
    count+=1
    #img=np.flip(img,axis=-1)
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_red=np.array([0,120,50])
    upper_red=np.array([9,255,255])
    mask1=cv2.inRange(hsv,lower_red,upper_red)
    
    lower_red=np.array([171,120,50])
    upper_red=np.array([180,255,255])
    mask2=cv2.inRange(hsv,lower_red,upper_red)
    
    mask1=mask1+mask2
    mask1=cv2.morphologyEx(mask1, cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask1=cv2.morphologyEx(mask1, cv2.MORPH_DILATE,np.ones((3,3),np.uint8))

    mask2=cv2.bitwise_not(mask1)
    cv2.imshow('Mask Vdo', mask2)
    res1=cv2.bitwise_and(img,img,mask=mask2)
    cv2.imshow("res1",res1)
    res2=cv2.bitwise_and(background,background,mask=mask1)
    
    finalOutput=cv2.addWeighted(res1,1,res2,1,0)
    
    cv2.imshow("magic",finalOutput)
    cv2.waitKey(10)


cap.release()
cv2.destroyAllWindows()

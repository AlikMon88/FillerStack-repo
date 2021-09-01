## Large number of colors can be detected simulteneously using distributed computing --
import cv2 as cv
import numpy as np
import ray

ray.init()

@ray.remote
def redDetect(hsv):
    lower_red=np.array([0,120,70]) #Hue-Color, Saturation-Shade_of_color, Value-Intensity
    upper_red=np.array([10,255,255])
    mask1=cv.inRange(hsv,lower_red,upper_red)
    
    lower_red=np.array([170,120,70])
    upper_red=np.array([180,255,255])
    mask2=cv.inRange(hsv,lower_red,upper_red)
    mask=mask1+mask2
    return mask

@ray.remote
def blueDetect(hsv):
    lower_blue = np.array([100,150,0])
    upper_blue = np.array([140,255,255])
    mask=cv.inRange(hsv,lower_blue,upper_blue)
    return mask

path='2cars.jpg'
path_1 = 'flower.jpg'
img=cv.imread(path)
img=cv.resize(img,(500,500))
cv.imshow('Actual', img)

hsv=cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
mask_id1 = redDetect.remote(hsv)
mask_id2 = blueDetect.remote(hsv)
        
mask1, mask2 = ray.get([mask_id1, mask_id2])
mask = mask1 + mask2
cv.imshow('HSV',hsv)

cv.imshow('Mask',mask)

mask1 = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3,3),np.uint8))
mask1 = cv.morphologyEx(mask, cv.MORPH_DILATE, np.ones((3,3),np.uint8))

#inverting mask
mask2=cv.bitwise_not(mask1)

res1=cv.bitwise_and(img,img,mask=mask2)
cv.imshow('Mask Image',res1)

img_cv=cv.imread(path_1)
img_cv=cv.resize(img_cv,(img.shape[0],img.shape[1]))


res2=cv.bitwise_and(img_cv,img_cv,mask=mask1)
final_op=cv.addWeighted(res1,1,res2,1,0)
cv.imwrite(r'C:\Users\User\Desktop\Final_img.jpg', final_op)

cv.imshow('Final', final_op)


cv.waitKey(0)
cv.destroyAllWindows()

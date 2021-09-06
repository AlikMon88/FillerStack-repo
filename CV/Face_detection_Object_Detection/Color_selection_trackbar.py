import cv2 as cv

def empty(a):
    pass

cv.namedWindow("Hsv")
cv.createTrackbar("Hue_min","Hsv",0,179,empty)
cv.createTrackbar("Hue_max","Hsv",179,179,empty)
cv.createTrackbar("Sat_min","Hsv",0,255,empty)
cv.createTrackbar("Sat_max","Hsv",255,255,empty)
cv.createTrackbar("Val_min","Hsv",0,255,empty)
cv.createTrackbar("Val_max","Hsv",255,255,empty)

img_path = r"F:\YT videos\park_seg.jpg"
img = cv.imread(img_path)
img = cv.resize(img,(500,500))

while True:
    hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)       
    h_min=cv.getTrackbarPos("Hue_min","Hsv")
    h_max=cv.getTrackbarPos("Hue_max","Hsv")
    s_min=cv.getTrackbarPos("Sat_min","Hsv")
    s_max=cv.getTrackbarPos("Sat_max","Hsv")
    v_min=cv.getTrackbarPos("Val_min","Hsv")
    v_max=cv.getTrackbarPos("Val_max","Hsv")
    print(h_min,h_max,s_min,s_max,v_min,v_max)
    
    lower_val = np.array([h_min,s_min,v_min])
    upper_val = np.array([h_max,s_max,v_max])
    mask=cv.inRange(hsv,lower_val,upper_val)
    

    cv.imshow('IMG',img)
    cv.imshow('mask',mask)
    if cv.waitKey(10) & 0xFF==ord('d'):
        break
cv.waitKey(1)
cv.destroyAllWindows()


cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()

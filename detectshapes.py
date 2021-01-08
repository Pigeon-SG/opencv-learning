import cv2
import numpy as np
import imagestack
def empty(a):
    pass

img = cv2.imread("pics/cards.jpg")
img0 = img
cv2.namedWindow("bars")
cv2.resizeWindow("bars",640,280)

cv2.createTrackbar("Blur","bars",0,20,empty)
cv2.createTrackbar("Canny","bars",0,1000,empty)

while True:
    blurvalue = cv2.getTrackbarPos("Blur","bars")
    cannyvalue = cv2.getTrackbarPos("Canny","bars")

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(7,7),blurvalue)
    imgCanny = cv2.Canny(imgBlur,cannyvalue,cannyvalue)

    contours,hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    image = imagestack.stackImages(0.6,([img,imgGray],[imgBlur,imgCanny]))
    image0 = cv2.drawContours(img0,contours,0,(0,0,255),3)

    cv2.imshow("2",image0)
    cv2.imshow("1",image)
    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyAllWindows()
        break
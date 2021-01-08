import cv2

img = cv2.imread("pics/sun.jpg")
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
area = imgGray[0:500,0:300]
imgBlur = cv2.GaussianBlur(imgGray,(11,11),0)

area_blur = cv2.GaussianBlur(area,(11,11),0)
imgGray[0:500,0:300] = area_blur
cv2.imshow("11",imgGray)
cv2.waitKey(0)
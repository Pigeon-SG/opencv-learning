import cv2
import numpy as np
import imagestack
def empty(a):
    pass
# 检测颜色
cv2.namedWindow("track bars")
cv2.resizeWindow("track bars",640,240)

img = cv2.imread("pics/pigeon.jpg")
imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# 确定阈值范围
# 创建一个track bar
cv2.createTrackbar("R_min","track bars",0,255,empty)
cv2.createTrackbar("G_min","track bars",0,255,empty)
cv2.createTrackbar("B_min","track bars",0,255,empty)
cv2.createTrackbar("R_max","track bars",255,255,empty)
cv2.createTrackbar("G_max","track bars",255,255,empty)
cv2.createTrackbar("B_max","track bars",255,255,empty)

while True:
    red_min = cv2.getTrackbarPos("R_min","track bars")
    green_min = cv2.getTrackbarPos("G_min","track bars")
    blue_min = cv2.getTrackbarPos("B_min","track bars")

    red_max = cv2.getTrackbarPos("R_max", "track bars")
    green_max = cv2.getTrackbarPos("G_max", "track bars")
    blue_max = cv2.getTrackbarPos("B_max", "track bars")

    lower = np.array([red_min,green_min,blue_min])
    upper = np.array([red_max,green_max,blue_max])
    mask = cv2.inRange(img,lower,upper)
    mask2 = cv2.bitwise_not(mask)
    imgresult = cv2.bitwise_and(img,img,mask=mask)
    # 利用 HSV 可以提取出颜色，使用原始的 RBG 提取不出来。。。

    imgstack = imagestack.stackImages(0.5,([img,imgHSV],[mask,imgresult]))
    cv2.imshow("1",imgstack)
    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyAllWindows()
        break




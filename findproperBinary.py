import cv2
import matplotlib.pyplot as plt
import numpy
import os
import paddlehub as hub
import Pigeon_utils as pigeon
def empty(a):
    pass
img = cv2.imread("pics/wuxian1.jpg")
human_seg = hub.Module(name="humanseg_mobile")
human_segmention = human_seg.segment(images=[img], visualization=False)
ret = human_segmention[0]["data"]
cv2.namedWindow("bars")
cv2.resizeWindow("bars",640,280)

cv2.createTrackbar("Binary","bars",194,255,empty)
while True:
    thresh = cv2.getTrackbarPos("Binary","bars")
    result,image_mask = cv2.threshold(ret,thresh,255,cv2.THRESH_BINARY)
    image_aftermask = cv2.bitwise_and(img,img,mask=image_mask)
    image_ret = pigeon.stackImages(0.5,[[img,image_mask,image_aftermask]])
    cv2.imshow("1",image_ret)
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        break
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import emoji_utils

margin_pos = []
def get_marginpos(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ret = [x,y]
        margin_pos.append(ret)
        print(margin_pos)

cv2.namedWindow("image")
cv2.namedWindow("face")
cv2.setMouseCallback("image",get_marginpos)
# 获取表情包的背景图片，并进行二值化处理
emoji_back = cv2.imread("pics/back.jpg")
ret,emoji_back = cv2.threshold(emoji_back,100,255,cv2.THRESH_BINARY)
face = cv2.imread("pics_processed/20210205/wyh_face_binary.jpg")

# 获取填充的坐标

cv2.imshow("image",face)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyWindow("image")

margin_area = np.array([margin_pos])
image_margin = cv2.fillPoly(face,[margin_area],(255,255,255))

cv2.imshow("face",image_margin)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyWindow("face")




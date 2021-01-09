import cv2
import numpy as np
from matplotlib import pyplot as plt
# 将方块提取出来

img = cv2.imread("pics/cards.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# 定义提取后的长宽
width,height = 500,700
points1 = np.float32([[633,71],[353,296],[510,498],[793,270]])
points2 = np.float32([[0,0],[0,height],[width,height],[width,0]])

#定义透视变化矩阵，各个点都要对齐
#类似于旋转加scale？
matrix1 = cv2.getPerspectiveTransform(points1,points2)
#这个和表情包制作里的旋转函数差不多，图像 + 映射矩阵
imgout = cv2.warpPerspective(img,matrix1,(width,height))

plt.imshow(imgout)
plt.show()

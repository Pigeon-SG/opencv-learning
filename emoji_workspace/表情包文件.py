import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
def plt_show(img):
    imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
image_path = 'pics_processed/xx_face2.jpg'
image = cv2.imread(image_path)    #imread 读进来是BGR格式数据
plt_show(image)
#   二值化处理（熊猫表情包必备）
ret,image_binary = cv2.threshold(image,100,255,cv2.THRESH_BINARY)
#旋转一下
    # 旋转需要搭配两个函数
    # mat  是旋转镜像矩阵
image_rotate_mat = cv2.getRotationMatrix2D((100,100),10,1)
image_rotate = cv2.warpAffine(image_binary,image_rotate_mat,(image_binary.shape[1],image_binary.shape[0]))

# 需要缩放成300 x 300 的矩阵
image_resize = cv2.resize(image_rotate, None, fx=1.5, fy=1.5, interpolation = cv2.INTER_CUBIC)
plt_show(image_resize)

#去除不必要的黑色
image_copy = image_resize.copy()
margin_area = np.array([[0,160],[50,75],[100,50],[150,0], #left
                  [200,60],[230,82],[240,200],
                  [200,270],[150,298],
                  [0,150],
                 [0,298],[298,298],[298,0],[0,0]
                  ])
image_margin = cv2.fillPoly(image_copy,[margin_area],(255,255,255))
face = image_margin[50:298,50:250,:]
plt_show(face)
#整合背景和表情
back = cv2.imread("pics/back.jpg")
#处理背景（同样的二值化处理）
ret,background = cv2.threshold(back,100,255,cv2.THRESH_BINARY)

h_xx,w_xx,z = face.shape
h_b,w_b,z = back.shape
    #利用矩阵覆盖掉背景中脸部的部分
left = (w_b - w_xx)//2
right = left + w_xx
top = 75
bottom = top + h_xx
emoji = background
emoji[top: bottom, left: right,:] = face
plt_show(emoji)
cv2.imwrite("pics_processed/xx_face_nonetext.jpg",emoji)
#添加中文文本
def emoji_addtext(emoji,text):
    # cv2 读取处理过的emoji图片
    Pilimg = Image.fromarray(emoji)
    draw = ImageDraw.Draw(Pilimg)
    font = ImageFont.truetype('simhei.ttf',60)
    # 文字居中处理 X轴向居中
    fsize = font.getsize(text)
    text_anchor_y = 450
    text_anchor_x = (emoji.shape[1] - fsize[0])//2
    draw.text((text_anchor_x,text_anchor_y),text,font=font,fill=0)
    emoji_text = cv2.cvtColor(np.array(Pilimg),cv2.COLOR_RGB2BGR)
    plt_show(emoji_text)
    return emoji_text

xx_face = emoji_addtext(emoji,"晚上几点钟？")
cv2.imwrite("my_emoji/xx_xm_face.jpg",xx_face)
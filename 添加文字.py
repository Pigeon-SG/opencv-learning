import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def plt_show(img):
    imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(imageRGB)
    plt.show()

def emoji_addtext(emoji,text):
    # cv2 读取处理过的emoji图片
    Pilimg = Image.fromarray(emoji)
    draw = ImageDraw.Draw(Pilimg)
    font = ImageFont.truetype('simhei.ttf',60)
    # 文字居中处理
    fsize = font.getsize(text)
    #       Y轴居中
    text_anchor_y = emoji.shape[0] - 50 - fsize[1]//2
    #       X轴向居中
    text_anchor_x = (emoji.shape[1] - fsize[0])//2
    draw.text((text_anchor_x,text_anchor_y),text,font=font,fill=0)
    emoji_text = cv2.cvtColor(np.array(Pilimg),cv2.COLOR_RGB2BGR)
    plt_show(emoji_text)
    return emoji_text
# CV2读取未添加过文字的图片

img = cv2.imread("pics_processed/wyh_emoji.jpg")
img_text = emoji_addtext(img,"今晚怎么说？")
ret = cv2.cvtColor(img_text,cv2.COLOR_BGR2RGB)
cv2.imwrite("my_emoji/wyhgezi.jpg",ret)
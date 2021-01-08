import cv2
import imagestack
import matplotlib.pyplot as plt
import numpy as np
import Pigeon_utils as pigeon
def plt_show(img):
    imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(imageRGB)
    plt.show()
face_contour = cv2.imread("pics_processed/wyh.jpg")
face_contour = cv2.cvtColor(face_contour,cv2.COLOR_RGB2GRAY)
#face_contour_binary 得到了汪雨涵的头像轮廓
ret,face_contour_binary = cv2.threshold(face_contour,100,255,cv2.THRESH_BINARY)
ret,face_contour_inv = cv2.threshold(face_contour,100,255,cv2.THRESH_BINARY_INV)


#resize 至符合嵌入的大小
face_contour_resize = cv2.resize(face_contour_binary,(face_contour_binary.shape[1]//6,face_contour_binary.shape[0]//6))
face_contour_inv_resize = cv2.resize(face_contour_inv,(face_contour_binary.shape[1]//6,face_contour_binary.shape[0]//6))

face = cv2.imread("pics/wyh.jpg")

face_withoutback = cv2.bitwise_and(face,face,mask=face_contour_binary)
face_withoutback = cv2.resize(face_withoutback,(face_withoutback.shape[1]//6,face_withoutback.shape[0]//6))

plt_show(face_withoutback)

# 背景加载
back = cv2.imread("pics/pigeon.jpg")
#制造 back_mask
back_mask = np.zeros_like(cv2.cvtColor(back,cv2.COLOR_BGR2GRAY))
center_pts = (70,169)
back_mask = pigeon.pic_insert(back_mask,face_contour_resize,center_pts)

face_fromback = pigeon.pic_fetch(back,face_contour_resize,center_pts)
face_fromback = cv2.bitwise_and(face_fromback,face_fromback,mask=face_contour_inv_resize)

emoji_wyh_face = cv2.add(face_withoutback,face_fromback)

emoji_wyh = pigeon.pic_insert(back,emoji_wyh_face,center_pts)
# 加入空白

padding = np.zeros_like(emoji_wyh)
padding[:] = 255
padding = cv2.resize(padding,(padding.shape[1],100))
ret = np.vstack([emoji_wyh,padding])
plt_show(ret)

cv2.imwrite("pics_processed/wyh_emoji.jpg",ret)
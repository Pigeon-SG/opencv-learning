from Pigeon_utils import *
img = cv2.imread("pics/wyh.jpg")
img = pic_resizebytimes(img,1/6.0)
mask,mask_inv = get_contour_outlines(img)
ret = stackImages(1,[[img,mask_inv,mask]])
#取出合适大小的框
back = cv2.imread("pics/pigeon.jpg")
center_pts = (70, 169)
box_fromback = pic_fetch(back,img,center_point=center_pts,scale=1.0)
#制作背景和头像，进行相加
ret = embedding_face(box_fromback,img,mask,mask_inv)
ret = pic_insert(back,ret,center_pts,1)
ret = add_textmargin(ret,"今晚啥时候？")
ret = cv2.cvtColor(ret,cv2.COLOR_BGR2RGB)
plt_show(ret)
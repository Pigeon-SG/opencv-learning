from Pigeon_utils import *

        #可以封装成 从一张图片抠出一个人来

back = cv2.imread("my_emoji/wyhgezi2.0.jpg")
#从图中获取目标


person_img_path  = "pics/wuxian1.jpg"
xx_person,box_person_coords,points = get_personfrompic(person_img_path)
#轴对称图片
xx_person = cv2.flip(xx_person,1)
#read contours big
bigcontour = cv2.imread("pics_processed/xx_qiche.jpg")
#获取 嵌入的图像
# 1、需要进行放缩以及反转
    #本次设计不需要放缩了
# 2、从背景中获取相应的位置

#get contour from big contour
xx_contour = pic_fetchbybox(bigcontour,box_person_coords,points)

#get contour and contourinv respondingly
xx_contour_horizontal = cv2.flip(xx_contour,1)
xx_contour_horizontal = cv2.cvtColor(xx_contour_horizontal,cv2.COLOR_BGR2GRAY)
rrrr,xx_contour_horizontal_inv = cv2.threshold(xx_contour_horizontal,100,255,cv2.THRESH_BINARY_INV)

rrr,xx_contour_horizontal = cv2.threshold(xx_contour_horizontal,100,255,cv2.THRESH_BINARY)
# embedding center coordinate point
center_pts = (188,256)
# get embedding box from back
xx_back = pic_fetch(back,xx_contour,center_pts)
# masking pics


emoji_box = embedding_face(xx_back,xx_person,xx_contour_horizontal,xx_contour_horizontal_inv)
emoji = pic_insert(back,emoji_box,center_point=center_pts)
cv2.imwrite("my_emoji/xxqigezi1.0.jpg",emoji)
plt_show(emoji)

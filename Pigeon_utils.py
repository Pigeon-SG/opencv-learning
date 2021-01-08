import cv2
import matplotlib.pyplot as plt
import numpy as np
import paddlehub as hub
import os

def plt_show(img):
    imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(imageRGB)
    plt.show()
def pic_insert(back_img,object_img,center_point,scale = 1):
    # center 就按照 cv 存贮的标准来
    back_insert_object = back_img
    object_resize = cv2.resize(object_img,(int(object_img.shape[1]/scale),int(object_img.shape[0]/scale)))
    # 获取替代框坐标
    height_min = int(center_point[0] - object_resize.shape[0]/2)
    height_max = int(center_point[0] + object_resize.shape[0]/2)
    width_min = int(center_point[1] - object_resize.shape[1]/2)
    width_max = int(center_point[1] + object_resize.shape[1]/2)
    back_insert_object[height_min:height_max,width_min:width_max] = object_resize

    return back_insert_object

def pic_fetch(back_img,object_img,center_point,scale = 1):
    # 根据输入的 目标图像大小，从背景中抠出相应大小的图像
    # center 就按照 cv 存贮的标准来
    back_fetch_object = back_img
    object_resize = cv2.resize(object_img, (int(object_img.shape[1] / scale), int(object_img.shape[0] / scale)))
    # 获取替代框坐标
    height_min = int(center_point[0] - object_resize.shape[0] / 2)
    height_max = int(center_point[0] + object_resize.shape[0] / 2)
    width_min = int(center_point[1] - object_resize.shape[1] / 2)
    width_max = int(center_point[1] + object_resize.shape[1] / 2)

    result = np.zeros_like(object_resize)
    result = back_img[height_min:height_max,width_min:width_max]
    return result

def get_contour_outlines(img):
    human_seg = hub.Module(name="humanseg_mobile")
    pass

if __name__ == "__main__":
    back = np.zeros((500,500),np.uint8)
    pic = np.zeros((50,50),np.uint8)
    pic[:] = 255
    ret = pic_insert(back,pic,(200,200),scale=1.5)
    cv2.imshow("1",ret)
    cv2.waitKey(0)
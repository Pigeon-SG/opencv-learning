import paddlehub as hub
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Pigeon_utils import plt_show,stackImages,empty
import os
'''
    希望输入一个图片，可以得到这个图片关于人像的分割，
    在根据人像的分割获取灰度或者其他的东西
    
    期望的命名规则：
        在pics文件夹下是未处理过的文件
        在pics_processed的文件是处理过的文件
            按照处理目的来命名文件，例如在pics文件夹中是xx.jpg
                人脸识别后 xx_face.jpg (pics_processed/xx_face.jpg)
                人体分割后 xx_humanseg.jpg
'''
# matplotlib.use("TkAgg")

def get_human_seg(img,get_path):
    # img 是 cv2 读取的图像矩阵
    # 利用paddle预训练模型处理图像
    # 得到的图像
    if os.path.isfile(get_path) == False:
        human_seg = hub.Module(name="humanseg_mobile")
        human_segmention = human_seg.segment(images=[img], visualization=False)
        ret = human_segmention[0]["data"]
        cv2.imwrite(get_path,ret)
        return cv2.imread(get_path)
    else:
        return cv2.imread(get_path)
def get_face_and_pos(img):
    # 目的是为了获得图像中的人脸坐标，以便获取人脸图像
    face_detector = hub.Module(name="pyramidbox_face_detection")
    face_detection = face_detector.face_detection(images=[img])
    pos = face_detection[0]["data"][0]
    pos["left"] = int(pos["left"])
    pos["top"] = int(pos["top"])
    pos["right"] = int(pos["right"])
    pos["bottom"] = int(pos["bottom"])
    img_face = img_primary[pos["top"]:pos["bottom"], pos["left"]:pos["right"]]
    return img_face,pos

def getproperyBinarythresh(img):
    cv2.namedWindow("bars")
    cv2.resizeWindow("bars", 640, 280)
    cv2.createTrackbar("Binary", "bars", 39, 255, empty)
    cv2.createTrackbar("Kernel", "bars", 1, 10, empty)
    # 灰度转化
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 腐蚀kernel
    erode_kernel = np.ones((5,5),np.uint8)
    # 膨胀kernel
    dilate_kernel = np.ones((5,5),np.uint8)
    thresh = 0
    while True:
        thresh = cv2.getTrackbarPos("Binary", "bars")
        kernel_size = cv2.getTrackbarPos("Kernel","bars")
        erode_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result, image_binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        image_erosion = cv2.erode(image_binary,erode_kernel,iterations=1)
        image_dilation = cv2.dilate(image_erosion,dilate_kernel,iterations=1)
        image_ret = stackImages(1.5, [[img, image_binary],[image_erosion,image_dilation]])
        cv2.imshow("get proper binary thresh",image_ret)
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break
    return thresh

if __name__ == "__main__":
    img_primary = cv2.imread("pics/wyh3.jpg")
    processed_path = "pics_processed/wyh3_segman.jpg"
    img_face,face_pos = get_face_and_pos(img_primary)
    thresh = getproperyBinarythresh(img_face)
    print(thresh)
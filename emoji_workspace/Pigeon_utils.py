import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import paddlehub as hub

from PIL import Image, ImageDraw, ImageFont
import os
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
    return np.array(Pilimg)
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
def plt_show(img):
    imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(imageRGB)
    plt.show()

def pic_resizebytimes(img,times):
    ret = cv2.resize(img,(int(img.shape[1] * times),int(img.shape[0] * times)))
    return ret


def pic_insert(back_img,object_img,center_point,scale = 1):
    #根据输入object_img的大小，插入背景back_img中
    #center_point 用于计算插入坐标
    #scale 缩小倍数
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
    # 根据输入object_img目标图像大小，从背景back_img中抠出相应大小的图像
    # center_point 用于计算抠图的坐标位置
    # scale 缩小倍数
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

def pic_fetchbybox(back_img,box,center_point):
    # 根据输入的捕获框box的大小，从对应图像（back_img）中获取对应位置（center_point）的图像
    # box 坐标按照
    #   0：左上角x坐标
    #   1：左上角y坐标
    #   2：右下角x坐标
    #   3：右下角y坐标
    # center_point 按照坐标 x,y
    height_min = int(center_point[1] - (box[3] - box[1]) / 2)
    height_max = int(center_point[1] + (box[3] - box[1]) / 2)
    width_min = int(center_point[0] - (box[2] - box[0]) / 2)
    width_max = int(center_point[0] + (box[2] - box[0]) / 2)

    ret = back_img[height_min:height_max,width_min:width_max]
    return ret

matplotlib.use("TkAgg")

def get_contour_outlines(img):
    human_seg = hub.Module(name="humanseg_mobile")
    res = human_seg.segment(images=[img], visualization=False)
    img_seg = res[0]["data"]
    # 获取正反面膜
    ret, face_contour = cv2.threshold(img_seg, 100, 255, cv2.THRESH_BINARY)
    ret, face_contour_inv = cv2.threshold(img_seg, 100, 255, cv2.THRESH_BINARY_INV)
    return face_contour,face_contour_inv

def embedding_face(box_fromback,face_img,mask,mask_inv):
    # 取出合适大小的嵌入框作为嵌入的小目标
    back = cv2.bitwise_and(box_fromback,box_fromback,mask = mask_inv)
    face = cv2.bitwise_and(face_img,face_img,mask=mask)
    box_processed = cv2.add(back,face)
    return box_processed

def add_textmargin(img,text):
    padding = np.zeros_like(img)
    padding[:] = 255
    padding = cv2.resize(padding, (padding.shape[1], 100))
    ret = np.vstack([img, padding])
    emoji = emoji_addtext(ret,text)
    return emoji

def getandsave_humanseg(img,save_path):
    human_seg = hub.Module(name="humanseg_mobile")
    human_segmention = human_seg.segment(images=[img], visualization=False)
    ret = human_segmention[0]["data"]
    ret_thresh,img = cv2.threshold(ret,190,255,cv2.THRESH_BINARY)
    cv2.imwrite(save_path,img)
    return img

def get_personfrompic(img_path):
    person = cv2.imread(img_path)
    object_detector = hub.Module(name='yolov3_resnet50_vd_coco2017')
    result = object_detector.object_detection(images=[person], visualization=False)
    box_coords = result[0]["data"]
    # 存储了一张图片中一个人的坐标
    box_person_coords = np.array([0, 0, 0, 0])
    box_num = len(result[0]["data"])
    for i in range(box_num):
        if box_coords[i]["label"] == "person":
            box_person_coords[0] = box_coords[i]['left']
            box_person_coords[1] = box_coords[i]['top']
            box_person_coords[2] = box_coords[i]['right']
            box_person_coords[3] = box_coords[i]['bottom']
    points = [(box_person_coords[2] + box_person_coords[0]) // 2, (box_person_coords[3] + box_person_coords[1]) // 2]
    person_fetch = pic_fetchbybox(person, box_person_coords, points)
    return person_fetch,box_person_coords,points

if __name__ == "__main__":
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


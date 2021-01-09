import cv2
import os
import paddlehub as hub

human_seg = hub.Module(name="humanseg_mobile")
img = cv2.imread("pics/wyh.jpg")

dir_path = "pics/"
images_path = [dir_path + i for i in os.listdir(dir_path)]
input_dict = {"image":images_path}

res = human_seg.segment(images = [img],visualization = True)
cv2.imwrite("pics_processed/wyh.jpg",res[0]['data'])
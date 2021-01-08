import paddle.fluid
import paddlehub as hub
import os
from matplotlib import pyplot as plt
import cv2

#1.导入模块

#2.加载模型
humanseg = hub.Module(name="deeplabv3p_xception65_humanseg")

#3.获取文件
dir_path = "pics/"
images_path = [dir_path + i for i in os.listdir(dir_path)]
input_dict = {"image":images_path}

#4.抠图
results = humanseg.segmentation(data = input_dict,output_dir = "kouren")
print(results)

import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def frame_addtext(img,text,size):
	# cv2 读取处理过的emoji图片
	Pilimg = Image.fromarray(img)
	draw = ImageDraw.Draw(Pilimg)
	# 字体的高度会随着size大小而变化，size = 60 的时候， 高度为58
	font = ImageFont.truetype('simhei.ttf', size)
	# 文字居中处理
	fsize = font.getsize(text)
	text_anchor_y = img.shape[0]  - fsize[1]//2 - 58
	text_anchor_x = (img.shape[1] - fsize[0]) // 2
	# text_anchor_x,y 获得左上角的坐标，现在每个字体的长宽为 60 X 58
	color_list = ["red","blue","yellow","pink","black","orange","green"]
	text_list = list(text)
	for i in range(len(text_list)):
		draw.text((text_anchor_x + i*size, text_anchor_y), text_list[i], font=font, fill=color_list[i])
	# draw.text((text_anchor_x, text_anchor_y), text, font=font, fill=0)
	emoji_text = cv2.cvtColor(np.array(Pilimg), cv2.COLOR_RGB2BGR)

	# plt_show(emoji_text)
	print(text_anchor_y,text_anchor_x,img.shape,fsize)
	return emoji_text

def read_video(video_path):
	video_cap = cv2.VideoCapture(video_path)
	# 参数为0的时候，打开第一个摄像头，参数是路径，就打开路径的视频文件
	frame_count = 0
	all_frames = []
	while True:
		ret, frame = video_cap.read()
		# read() 是按照帧来读取视频文件， ret读取帧正确的判断，frame返回的是这个帧的图片

		if ret is False:
			break
		if frame_count > 25:
			if (frame_count % 4) > 2:
				frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			frame_text = frame_addtext(frame,"她微笑",60)
		else:
			frame_text = frame_addtext(frame,"As丶各种变态",40)
		all_frames.append(frame_text)
		cv2.imshow('frame', frame)
		cv2.waitKey(1)
		frame_count += 1
		print(frame_count)
	video_cap.release()
	cv2.destroyAllWindows()
	print('===>', len(all_frames))

	return all_frames


def frame_to_gif(frame_list):
	gif = imageio.mimsave('./pure_exp04.gif', frame_list, 'GIF',duration = 0.05)  # 0.05


if __name__ == "__main__":
	frame_list = read_video('pure.gif')
	frame_to_gif(frame_list)


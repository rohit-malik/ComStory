from PIL import Image
import numpy as np
import sys
def choose_col(dic,key):
	lst = dic[key]
	prob_list = []
	color_list = []
	for each in lst:
		prob,color = each.split(',')
		prob_list.append(prob)
		color_list.append(color)
	return np.random.choice(color_list, 1,p=prob_list)[0]


def get_one_frame(emotion):
	return Image.new("RGB", (1000, 500), choose_col(emotion_dict,emotion))


def stack_frames(lst):
	images = lst
	widths, heights = zip(*(i.size for i in images))
	total_height = sum(heights)
	max_width = max(widths)
	new_im = Image.new('RGB', (max_width, total_height))

	y_offset = 0
	for im in images:
	  new_im.paste(im, (0,y_offset))
	  y_offset += im.size[0]
	new_im.save('test.jpg')

emotion_dict = {'Angry': ['0.7,#FF0000','0.3,#d3d3d3'], 'Sad': ['0.1,#800080','0.3,#0d98ba','0.6,#8a2be2'], 'Neutral': ['1.0,#008000'], 'Disgust': ['0.7,#99ff33','0.25#f2ba49','0.05#0d98ba'], 'Surprise': ['1.0,#ffa500'], 'Fear': ['0.20,#8a2be2','0.68,#666666','0.12,#d3d3d3'], 'Happy': ['0.07,#ff0000','0.27,#ffff00','0.10,#00ff00','0.03,#0000ff','0.08,#800080','0.12,#f2ba49','0.05,#adff2f','0.14,#0d98ba','0.05,#8a2be2','0.09,#c71585']}
#img = Image.new("RGB", (1000, 500), '#00ff00')
im1 = Image.new("RGB", (1000, 500), choose_col(emotion_dict,'Angry'))
im2 = Image.new("RGB", (1000, 500), choose_col(emotion_dict,'Sad'))
im3 = Image.new("RGB", (1000, 500), choose_col(emotion_dict,'Surprise'))
stack_frames([im1,im2,im3])
'''
vis = np.concatenate((im2, im3), axis=0)
img = Image.fromarray(vis, 'RGB')
vis = np.concatenate((im1, img), axis=0)
img = Image.fromarray(vis, 'RGB')
#img.show()
img.save('Frame_image.png')
#im.show()

#print(choose_col(emotion_dict,'Happy'))'''

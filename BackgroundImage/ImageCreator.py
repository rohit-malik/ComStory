from PIL import Image
import numpy as np
emotion_dict = {'Angry': '#FF0000', 'Sad': '#800080', 'Neutral': '#008000', 'Disgust': '#99ff33', 'Surprise': '#ffa500', 'Fear': '#8a2be2', 'Happy': '#0d98ba'}
im1 = Image.new("RGB", (1000, 500), emotion_dict['Angry'])
im2 = Image.new("RGB", (1000, 500), emotion_dict['Sad'])
im3 = Image.new("RGB", (1000, 500), emotion_dict['Surprise'])

vis = np.concatenate((im1, im2), axis=0)
img = Image.fromarray(vis, 'RGB')
vis = np.concatenate((im3, img), axis=0)
img = Image.fromarray(vis, 'RGB')
img.show()
img.save('angry_image.png')
#im.show()


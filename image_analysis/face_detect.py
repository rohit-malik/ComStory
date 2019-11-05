from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
import cv2


image1 = Image.open("040wrmpyTF5l.jpg")
image_array1 = np.array(image1)
#plt.imshow(image_array1)


#image = face_recognition.load_image_file("040wrmpyTF5l.jpg")
image = face_recognition.load_image_file("different_emotions2.jpg")
#image = face_recognition.load_image_file("sad4.jpg")



face_locations = face_recognition.face_locations(image)
print(face_locations)
face_image_list = []
for each in range(len(face_locations)):
	top, right, bottom, left = face_locations[each]
	face_image = image[top:bottom, left:right]
	#plt.imshow(face_image1)
	image_save = Image.fromarray(face_image)
	name = "image_" + str(each) + ".jpg"
	image_save.save(name)
	face_image_list.append(face_image)




#top, right, bottom, left = face_locations[1]
#face_image2 = image[top:bottom, left:right]
#plt.imshow(face_image2)
#image_save = Image.fromarray(face_image2)
#image_save.save("image_2.jpg")


emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}


for each in face_image_list:
	face_image = each
	face_image = cv2.resize(face_image, (48,48))
	face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
	face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])

	model = load_model("model_v6_23.hdf5")

	#print(face_image.shape)

	predicted_class = np.argmax(model.predict(face_image))
	label_map = dict((v,k) for k,v in emotion_dict.items()) 
	predicted_label = label_map[predicted_class]
	print(predicted_label)
	print("***********")



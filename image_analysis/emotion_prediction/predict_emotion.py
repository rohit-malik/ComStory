from EmoPy.src.fermodel import FERModel
from pkg_resources import resource_filename

'''
target_emotions = ['surprise', 'calm', 'fear', 'anger']
target_emotions = ['disgust', 'happiness', 'surprise']
target_emotions = ['surprise', 'fear', 'anger']
target_emotions = ['calm', 'fear', 'anger']
target_emotions = ['calm', 'happiness', 'anger']
target_emotions = ['disgust', 'fear', 'anger']
target_emotions = ['disgust', 'calm', 'surprise']
target_emotions = ['disgust', 'surprise', 'sadness']
target_emotions = ['happiness', 'anger']
target_emotions = ['happiness', 'anger','surprise','calm']
'''

mp = {}
weight = {}
target_emotions_list = [['surprise', 'calm', 'fear', 'anger'],['disgust', 'happiness', 'surprise'],['surprise', 'fear', 'anger'],['calm', 'fear', 'anger'],['calm', 'happiness', 'anger'],['disgust', 'fear', 'anger'],['disgust', 'calm', 'surprise'],['disgust', 'surprise', 'sadness'],['happiness', 'anger']]
for each in target_emotions_list:
	target_emotions = each#['disgust', 'surprise', 'sadness']
	model = FERModel(target_emotions, verbose=True)
	print('Predicting on image...')
	emotion,percentage = model.predict('image_13.jpg')
	if emotion not in mp.keys():
		mp[emotion] = percentage
		weight[emotion] = 1
	else:
		temp = mp[emotion]
		mp[emotion] = temp+percentage
		weight[emotion] = weight[emotion] + 1
print(mp)
print(weight)
for each in mp.keys():
	mp[each] = mp[each]/weight[each]
print(mp)


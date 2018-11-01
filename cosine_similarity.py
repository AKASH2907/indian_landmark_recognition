import math
import numpy as np
from os.path import isfile, join
from os import listdir 
import cv2
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

monuments = ["Buddhist", "Dravidian", "Kalinga", "Mughal"]
monuments_check = ["Buddhist"]

train_features_path = './train_data_features'
val_feature_path = './val_data_features'
test_feature_path = './test_data_features'

x_train_features = []

buddhist = np.load('Buddhist.npy')
buddhist = buddhist.reshape((512,1))

dravidian = np.load('Dravidian.npy')
dravidian = dravidian.reshape((512, 1))

kalinga = np.load('Kalinga.npy')
kalinga = kalinga.reshape((512, 1))

mughal = np.load('Mughal.npy')
mughal = mughal.reshape((512, 1))


distance = []

def cos_sim(a, b):
	"""Takes 2 vectors a, b and returns the cosine similarity according 
	to the definition of the dot product
	"""
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)



for i in monuments:
	monument = join(train_features_path, i)

	files = listdir(monument)
	samples = random.sample(files, 100)
	print(samples)
	a = []

	for sample in samples:
		l = np.load(join(monument, sample))
		l = l.reshape((1, 512))
		# print(l.shape)
		a+=[l]
	
	# print(a)
	a = np.asarray(a)
	# print(a)
	print(a.shape)
	# print(a[0])
	sums = np.sum(a, axis=0)
	print(sums[0][0])
	sums/=100
	print(sums[0][0])
	print(sums.shape)
	# np.save( i +'.npy', sums)


# for i in monuments:
# 	monument = join(test_feature_path, i)
# 	files = listdir(monument)

# 	for file in files:
# 		data = join(monument, file)
# 		dat = np.load(data)
# 		dat = dat.reshape((1, 512))
# 		# cosi = cosine_similarity(dat, buddhist)
# 		cosi_b = cos_sim(dat, buddhist)
# 		cosi_d = cos_sim(dat, dravidian)
# 		cosi_k = cos_sim(dat, kalinga)
# 		cosi_m = cos_sim(dat, mughal)
# 		distance +=[[cosi_b, cosi_d, cosi_k,cosi_m]]
# 		# print(cosi_b)

# distance = np.asarray(distance)
# # print(distance)
# print(distance.shape)
# distance = distance.reshape((351, 4))
# print(distance)
# print(distance.shape)

# np.save('Y_test_predictions_7.npy', distance)
# for i in range(5):
# 	sd = np.argmax(distance[i])
# 	print(sd)




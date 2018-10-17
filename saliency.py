import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from os.path import isfile, join
from os import rename, listdir, rename, makedirs

# img = "../final_data/train/blasti/100101.jpg"
# image = cv2.imread(img)

monuments = ["Buddhist", "Dravidian", "Kalinga", "Mughal"]

destination = '../cropped_images/'
datapath = '../train_data/'

ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
# 	help="path to BING objectness saliency model")	
ap.add_argument("-n", "--max-detections", type=int, default=3,
	help="maximum # of detections to examine")
args = vars(ap.parse_args())

# initialize OpenCV's objectness saliency detector and set the path
# to the input model files
saliency = cv2.saliency.ObjectnessBING_create()
# saliency.setTrainingPath(args["model"])
saliency.setTrainingPath('./ObjectnessTrainedModel')


for i in monuments:
	path = join(datapath, i)
	files = listdir(path)
	files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

	cropped_path = join(destination, i)
	j=0
	for image in files:
		im = join(path, image)
		read = cv2.imread(im, 1)

		# compute the bounding box predictions used to indicate saliency
		(success, saliencyMap) = saliency.computeSaliency(read)
		numDetections = saliencyMap.shape[0]

		# loop over the detections
		
		for i in range(0, min(numDetections, args["max_detections"])):
			# extract the bounding box coordinates
			
			(startX, startY, endX, endY) = saliencyMap[i].flatten()

			cropped_image = read[int(startY):int(endY), int(startX):int(endX)]

			if j<10:
				cv2.imwrite(cropped_path + '/' + image[0:2] + '000' + str(j) + '.jpg', cropped_image)
			elif 10<=j<=99:
				cv2.imwrite(cropped_path + '/' + image[0:2] + '00' + str(j) + '.jpg', cropped_image)
			elif 100<=j<=999:
				cv2.imwrite(cropped_path + '/' + image[0:2] + '0' + str(j) + '.jpg', cropped_image)
			elif j>=1000:
				cv2.imwrite(cropped_path + '/' + image[0:2] + str(j) + '.jpg', cropped_image)
			j+=1

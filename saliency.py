import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from os.path import isfile, join
from os import rename, listdir, rename, makedirs

# img = "../final_data/train/blasti/100101.jpg"
# image = cv2.imread(img)

monuments = ["Buddhist", "Dravidian", "Kalinga", "Mughal"]

# destination = '../crops_txt/'
datapath = '../train_data/'

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to BING objectness saliency model")	
ap.add_argument("-n", "--max-detections", type=int, default=20,
	help="maximum # of detections to examine")
args = vars(ap.parse_args())

# initialize OpenCV's objectness saliency detector and set the path
# to the input model files
saliency = cv2.saliency.ObjectnessBING_create()
saliency.setTrainingPath(args["model"])


for i in monuments:
	path = join(datapath, i)
	files = listdir(path)
	files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	# txt_path = join(datapath, i)
	for image in files:
		im = join(path, image)

		read = cv2.imread(im, 1)


		print(read.shape)
		# compute the bounding box predictions used to indicate saliency
		(success, saliencyMap) = saliency.computeSaliency(read)
		numDetections = saliencyMap.shape[0]

		# loop over the detections
		# points = []
		for i in range(0, min(numDetections, args["max_detections"])):
			# extract the bounding box coordinates
			
			(startX, startY, endX, endY) = saliencyMap[i].flatten()
			
			print(startX, startY, endX, endY)
			print(type(startX))
			# points+=[[startX, endX, startY, endY]]
			# print(points)
			# randomly generate a color for the object and draw it on the image
			output = read.copy()
			color = np.random.randint(0, 255, size=(3,))
			color = [int(c) for c in color]
			cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)


			cropped_image = read[int(startX):int(endX), int(startY):int(endY)]

			cv2.imshow("Image", cropped_image)
			cv2.waitKey(0)

		# print(points)

			# with open(txt_path+ '/'+ str(image[0:6]) + '.txt', 'a+') as f:
			# 	# for i in points:
			# 	f.write(str(startX) + " " + str(endX) + " " + str(startY) + " " + str(endY) +"\n")

		break
	# break
	# show the output image
	# cv2.imshow("Image", output)
	# cv2.waitKey(0)
	# plt.figure()
	# plt.imshow(output)
	# plt.show()

# cv2.destroyAllWindows()








# image = cv2.resize(image, (512, 512))
# saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
# (success, saliencyMap) = saliency.computeSaliency(image)
# saliencyMap = (saliencyMap * 255).astype("uint8")
# cv2.imshow("Image", image)
# cv2.imshow("Output", saliencyMap)
# cv2.waitKey(0)

# saliency = cv2.saliency.StaticSaliencyFineGrained_create()
# (success, saliencyMap) = saliency.computeSaliency(image)
 
# # if we would like a *binary* map that we could process for contours,
# # compute convex hull's, extract bounding boxes, etc., we can
# # additionally threshold the saliency map
# threshMap = cv2.threshold(saliencyMap, 0, 255,
# 	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
 
# # show the images
# # cv2.imshow("Image", image)
# cv2.imshow("Output", saliencyMap)
# cv2.imshow("Thresh", threshMap)
# cv2.waitKey(0)
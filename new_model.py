import sys
import random
import math
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout, Flatten, Input, AveragePooling2D, BatchNormalization
from keras.models import Model
from keras.utils import plot_model, np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from time import time
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import backend as K
from os.path import isfile, join
from os import rename, listdir, rename, makedirs
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.utils.generic_utils import get_custom_objects
from keras.regularizers import l2
import argparse

# model_iv3 = InceptionV3()

# model_ivr2 = InceptionResNetV2()

monuments = ["Buddhist", "Dravidian", "Kalinga", "Mughal"]
monuments_check = ["Buddhist"]

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--max-detections", type=int, default=3,
	help="maximum # of detections to examine")
args = vars(ap.parse_args())
saliency = cv2.saliency.ObjectnessBING_create()
saliency.setTrainingPath('./ObjectnessTrainedModel')



def build_inception_resnet_V2(img_shape=(416, 416, 3), n_classes=4, l2_reg=0.,
                load_pretrained=True, freeze_layers_from='base_model'):
    # Decide if load pretrained weights from imagenet
    if load_pretrained:
        weights = 'imagenet'
    else:
        weights = None

    # Get base model
    base_model = InceptionResNetV2(include_top=False, weights=weights,
                             input_tensor=None, input_shape=img_shape)

    # Add final layers
    x = base_model.output
    x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='dense_1', kernel_initializer='he_uniform')(x)
    x = Dropout(0.25)(x)
    predictions = Dense(n_classes, activation='softmax', name='predictions', kernel_initializer='he_uniform')(x)


    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
            print ('   Freezing base model layers')
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for i, layer in enumerate(model.layers):
                print(i, layer.name)
            print ('   Freezing from layer 0 to ' + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
               layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
               layer.trainable = True

    adam = Adam(0.0001) 
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])  
    # model.summary()
    return model


savepath = '../../../../'
savepath1 =join(savepath, '/mnt/data/rajiv/akash/akash/')

val_image_path = './val_data'

model = build_inception_resnet_V2()

# model.load_weights(savepath1 + 'wts/irv2_saliency/irv2_saliency-02-0.95.hdf5')
model.load_weights(savepath1 + 'wts/saliency+irv2/saliency+irv2-04-0.80.hdf5')

for i in monuments_check:

	monument = join(val_image_path, i)

	files = listdir(monument)

	print(len(files))

	for file in files:
		img_path = join(monument, file)

		image = cv2.imread(img_path, 1)

		print(image.shape)

		# break

		batches = []
		(success, saliencyMap) = saliency.computeSaliency(image)
		numDetections = saliencyMap.shape[0]

		for i in range(0, min(numDetections, args["max_detections"])):
			# extract the bounding box coordinates
			
			(startX, startY, endX, endY) = saliencyMap[i].flatten()
			# print(startX, startY, endX, endY)
			cropped_image = image[int(startY):int(endY), int(startX):int(endX)]
			cropped_image = cv2.resize(cropped_image, (416, 416))
			print(cropped_image.shape)
			batches.append(cropped_image)


		batches = np.asarray(batches).astype('float32')
		batches/=255
		# print(batches)
		print(batches.shape)

		predictions = model.predict(batches)

		print(predictions)
		break



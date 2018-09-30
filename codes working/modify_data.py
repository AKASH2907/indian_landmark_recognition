import cv2
import numpy as np
# from sklearn.model_selection import train_test_split
from os.path import isfile, join
from os import rename, listdir, rename, makedirs
import argparse
from shutil import copyfile, move
# import imgaug as ia
# from imgaug import augmenters as iaa
import random

# datapath = '../train_data/'
# datapath = '../train_data/'
monuments = ["Buddhist", "Dravidian", "Kalinga", "Mughal"]

# destination = '../train_data/'
datapath = './dataset'


train_list = []
def rename_files():
	classes = 1

	for i in monuments:
		path = join(datapath, i)
		files = listdir(path)
		files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
		# print(i)
		for file in files:
			# print(file)
			# train_list.append(file)
			dest = join(destination, i)
			# print(files)
			if classes<10:
				j = 0
				for file in files:
					if j<10:
						print(join(path, file))
						# 00 - Data Augmentation 00- Classes 00 - Images 000101.jpg
						copyfile(join(path, file), join(dest,  str(100) + str(classes) + str(0) + str(j) + '.jpg'))
						print(join(dest,  str(100) + str(classes) + str(0) + str(j) + '.jpg'))
					elif j>=10:
						print(join(path, file))
						copyfile(join(path, file), join(dest, str(100) + str(classes) + str(j) + '.jpg'))
						print(join(dest, str(100) + str(classes) + str(j) + '.jpg'))
					print(j)
					j+=1
			elif classes>=10:
				j = 0
				for file in files:
					if j<10:
						print(join(path, file))
						copyfile(join(path, file), join(dest, str(10) + str(classes) + str(0) + str(j) + '.jpg'))
						print(join(dest, str(10) + str(classes) + str(0) + str(j) + '.jpg'))
					elif j>=10:
						print(join(path, file))
						copyfile(join(path, file), join(dest, str(10) + str(classes) + str(j) + '.jpg'))
						print(join(dest, str(10) + str(classes) + str(0) + str(j) + '.jpg'))
					print(j)
					j+=1

		classes+=1
		# print(classes)
		# print(path)
		# break

	return train_list

def save_images(augmentated_image, k, classes):

    # im = cv2.resize(augmentated_image, (1000, 1000))
    if classes<10:
        if k<10:
            print(join(destination, str(1) + str(classes) + str(0) + str(k) + '.jpg'))
            cv2.imwrite(join(destination, str(1) + str(classes) + str(0) + str(k) + '.jpg'), im)
        elif k>=10:
            print(join(destination, str(1) + str(classes) + str(k) + '.jpg'))
            cv2.imwrite(join(destination, str(1) + str(classes) + str(k) + '.jpg'), im)

    elif classes>=10:
        if k<10:
            print(join(destination, str(classes) + str(0) + str(k) + '.jpg'))
            cv2.imwrite(join(destination, str(classes) + str(0) + str(k) + '.jpg'), im)

        elif k>=10:
            print(join(destination, str(classes) + str(k) + '.jpg'))
            cv2.imwrite(join(destination, str(classes) + str(k) + '.jpg'), im)

channel = 0
lengths = []
def read_images():
	channel=0
	for i in monuments:
		path = join(datapath, i)
		files = listdir(path)
		# lengths.append(len(files))
		print(len(files))
		channel+=len(files)
		print(channel)
		# files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
		# for image in files:
		# 	im = join(path, image)
		# 	# print(im)

		# 	img = cv2.imread(im, 1)
		# 	print(img.shape)



# train_lists = rename_files()	
# print(train_lists)
# np.save('../train_lists.npy', train_lists)
# read_images()
# print(lengths)
# np.save('lengths.npy', lengths)
# resize_images()
# augmentation()
# create_test()

path = './test_data'

for i in monuments:
	
	makedirs(join(path, i))































def resize_images():
	c = 1
	for i in monuments:
	    source = join(datapath, i)
	    files = listdir(source)
	    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	    j=0
	    for image in files:
	        # print(image)
	        im = join(source, image)

	        read = cv2.imread(im, 1 )
	        # print(read.shape)
	        # print(read)
	        save_images(read, j, c)
	        j+=1
	    c+=1





def create_test():
	for i in monuments:
		source = join(destination, i)
		final = join(path, i)
		files = listdir(source)
		number = len(files)
		# l = random.sample(files, 5)
		# print(l)
		# break
		# for f in files:

		# 	print(join(source, f))
		if(number<8):
			l = random.sample(files, 2)
			for f in l:
				move(join(source, f), join(final, f))

		elif number==9:
			l = random.sample(files, 3)
			for f in l:
				move(join(source, f), join(final, f))

		elif number==10:
			l = random.sample(files, 4)
			for f in l:
				move(join(source, f), join(final, f))

		elif 10<number<20:
			l = random.sample(files, 5)
			for f in l:
				move(join(source, f), join(final, f))

		elif number==20:
			l = random.sample(files, 6)
			for f in l:
				move(join(source, f), join(final, f))


def augmentation():
	c = 1
	for i in monuments:
		path = join(datapath, i)
		files = listdir(path)
		files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
		batches = []
		for image in files:
			images = np.array([cv2.imread(join(path, image), 1) for image in files], dtype=np.uint8)
		seq = iaa.Sequential([iaa.Fliplr(0.5)], random_order=True)
		images_aug = seq.augment_images(images)
		j =0
		for k in images_aug:
			save_images(k, j, c)
			j +=1
		c +=1
		break
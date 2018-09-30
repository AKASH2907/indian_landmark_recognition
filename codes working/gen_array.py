import cv2
from matplotlib import pyplot as plt
import numpy as np
from os.path import isfile, join
from os import rename, listdir, rename, makedirs

monuments = ["Buddhist", "Dravidian", "Kalinga", "Mughal"]

# new_species = 
datapath = './'
# sftp://rajivratn@matterhorn.d2.comp.nus.edu.sg/home/rajivratn/akash/akash/features_images_cropped
N_CLASSES = 4
def gen_data():
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    count=0
    for i in monuments:

        # train_samples = join(datapath, 'validation_datas/' + i)
        test_samples = join(datapath, 'tests/'+i)
        # train_files = listdir(train_samples)
        # print(train_files)
        test_files = listdir(test_samples)
        # train_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        test_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        # for j in train_files:
            # im = join(train_samples, j)
            # data = np.load(im)
            # print(data.shape)
            # img = cv2.imread(im,1)
            # print(im)
            # img = cv2.resize(img, (416, 416))
            # print(img)
            # data = np.reshape(data, (512, 1))
            # print(data.shape)
            # X_train.append(data)
            # Y_train+=[count]
            # break
        # print(count)

        for k in test_files:
            im = join(test_samples, k)
            img = cv2.imread(im,1)
            img = cv2.resize(img, (416, 416))
            # print(im)
            # data = np.load(im)
            # data = np.reshape(data, (512, 1))
            X_test.append(img)
            Y_test+=[count]
        

        count+=1
        # break

    # print(X_train)
    # print(X_train.shape)
    # print(X_train_batch)
    # print(len(X_train))

    # X_train = np.asarray(X_train)
    # X_train = X_train.astype('float32')
    # X_train/= 255
    # X_train = np.asarray(X_train)
    # Y_train = np.asarray(Y_train)
    # Y_train = np_utils.to_categorical(Y_train, len(monuments)-1)
    # print(X_train)
    # print(y_train)
    # print(arr)
    # print(arr.shape)
    # print(X_train.shape)
    # print(Y_train.shape)
    # print(y_train)
    # X_train = X_train.reshape(150, 416*416*3)

    X_test = np.asarray(X_test)
    X_test = X_test.astype('float32')
    X_test /= 255
    Y_test = np.asarray(Y_test)
    # Y_test = np_utils.to_categorical(Y_test, N_CLASSES)
    return X_test, Y_test

# savepath = '../../../../'
# savepath1 =join(datapath, '/mnt/data/rajiv/akash/akash')
x_train, y_train = gen_data()
# print(x_train.shape)
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# y_train = np_utils.to_categorical(y_train, N_CLASSES)
# x_train_features = np.reshape(x_train_features, (x_train_features.shape[0], 512))
# print(x_train_features.shape)
# np.save('X_train_siamese_testing.npy', x_train)
# np.save('Y_train_siamese_testing.npy', y_train)

# x_train, y_train, x_test, y_test = gen_data()
# y_train = np_utils.to_categorical(y_train, N_CLASSES)
# y_test = np_utils.to_categorical(y_test, N_CLASSES)
np.save('X_test.npy', x_train)
np.save('Y_test_not_categorical.npy', y_train)
# np.save('y_valid_new.npy', y_train)
# np.save('Y_validation.npy', y_train)
# np.save('X_test_features.npy', x_test)
# np.save("Y_test_features.npy", y_test) 
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# print(y_train, y_test)

print("Done... saved files")

# def gen_batch():
#     X_train = []
#     for i in monuments:

#         train_samples = join(datapath, 'train/'+i)
#         test_samples = join(datapath, 'test/'+i)
#         train_files = listdir(train_samples)
#         test_files = listdir(test_samples)
#         train_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
#         test_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

#         for j in train_files:
            

# y_test, y_train = gen_data()
# np.save('Y_test_not_categorical.npy', y_test)
# np.save('Y_train_not_categorical.npy', y_train)
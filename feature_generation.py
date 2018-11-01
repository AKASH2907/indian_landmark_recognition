import math
import numpy as np
from os.path import isfile, join
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten, Input, AveragePooling2D
from keras.optimizers import Adam
from keras.models import Model, model_from_json
from keras.utils import plot_model, np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from os import listdir 
import cv2
from keras.preprocessing.image import load_img, img_to_array


# monuments = ["Buddhist", "Dravidian", "Kalinga", "Mughal"]

# savepath = '../../../../'
# savepath2 = join(savepath, '/mnt/data/rajiv/akash/')
# # image_path = join(savepath2, 'test_data/')

# image_path = './train_data'
# save_path = './train_irv2'


# # model = InceptionV3()
# model = InceptionResNetV2()
# # re-structure the model
# model.layers.pop()
# model = Model(inputs=model.inputs, outputs=model.layers[-1].output)




# for i in monuments:
#     monument = join(image_path, i)

#     files = listdir(monument)
#     # files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

#     save_files = join(save_path, i)
#     # save_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
#     # st_path = join(string_path, i)
#     # str_files = listdir(st_path)
#     # str_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))    
#     c =0
#     for j in files:
#         str_name = j[:6]
#         path = join(monument, j)
#         image = load_img(path, target_size=(416, 416))
#         image = img_to_array(image)
#         image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#         image = preprocess_input(image)
#         # print(image)
#         feature = model.predict(image, verbose=0)

#         # img = cv2.imread(path, 1)
#         # img = img[..., ::-1]
#         # img = img.astype(np.float32)
#         # img = cv2.resize(img, (416, 416))

#         # img = np.reshape(img , (1, 416, 416 , 3))
#         # img = preprocess_input(img)
#         # print(img)
#         # preds= model.predict(img)

#         # preds_arr = np.asarray(preds)

#         # print(preds_arr)
#         # print(preds_arr.shape)
#         # preds_arr = preds_arr.reshape(512, 1)
#         # print(preds_arr.shape)
#         # print(str_name)
#         print(str_name)
#         np.save(save_files + '/' + str_name + '.npy', feature)

#     # print(c)
#     # c+=1
#         # break

#     # break















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

    # adam = Adam(0.0001) 
    # model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])  
    # model.summary()
    return model


savepath = '../../../../'
savepath1 =join(savepath, '/mnt/data/rajiv/akash/akash/')
# x_train = np.load(savepath1 + 'X_train_saliency_new.npy')
x_test = np.load(savepath1 +'X_test_new.npy')
x_valid = np.load(savepath1 +'X_valid_new.npy')
# y_train = np.load('Y_train_saliency_new.npy')
# y_valid = np.load('Y_valid.npy')
y_test = np.load('Y_test.npy')

model = build_inception_resnet_V2()

model.load_weights(savepath1 + 'wts/irv2+saliency/irv2+saliency_new-03-0.81.hdf5')

model = Model(inputs = model.input, outputs = model.get_layer('dense_1').output)
print(model.summary())
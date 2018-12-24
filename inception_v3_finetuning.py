import math
import numpy as np
from os.path import isfile, join
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten, Input, AveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import plot_model, np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.applications.inception_v3 import InceptionV3


BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
N_CLASSES = 4
EPOCHS = 7


def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Activation(swish)})


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 4.0
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate



def build_inceptionV3(img_shape=(512, 512, 3), n_classes=4, l2_reg=0.,
                load_pretrained=True, freeze_layers_from='base_model'):
    # Decide if load pretrained weights from imagenet
    if load_pretrained:
        weights = 'imagenet'
    else:
        weights = None

    # Get base model
    base_model = InceptionV3(include_top=False, weights=weights,
                             input_tensor=None, input_shape=img_shape)

    # Add final layers
    x = base_model.output
    x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='swish', name='dense_1', kernel_initializer='he_uniform')(x)
    x = Dropout(0.25)(x)
    predictions = Dense(n_classes, activation='softmax', name='predictions', kernel_initializer='he_uniform')(x)

    # This is the model we will train
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
x_train = np.load(savepath1 + 'X_train_512.npy')
x_test = np.load(savepath1 +'X_test_512.npy')
x_valid = np.load(savepath1 +'X_valid_512.npy')
y_train = np.load('Y_train.npy')
y_valid = np.load('Y_valid.npy')
y_test = np.load('Y_test.npy')

print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape, y_test.shape)

filepath = savepath1 + "wts/iv3/inception_v3_512-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose =1, save_best_only=True, mode='max', save_weights_only=True)
# tensorboard = TensorBoard(log_dir=log_path,
#                                 write_graph=False, #This eats a lot of space. Enable with caution!
#                                 #histogram_freq = 1,
#                                 write_images=True,
#                                 batch_size = BATCH_SIZE,
#                                 write_grads=True)
# lrate = LearningRateScheduler(step_decay)

callback = [checkpoint]


model = build_inceptionV3()

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=7, verbose= 1, 
    # steps_per_epoch=x_train.shape[0]//BATCH_SIZE,
    validation_data=(x_valid, y_valid),
    callbacks = callback
    )


# model.load_weights(savepath1 + 'wts/iv3/inception_v3-06-0.82.hdf5')
# model.load_weights(savepath1 + 'wts/iv3/inception_v3_512-04-0.78.hdf5')

score = model.evaluate(x_test, y_test, verbose=1, batch_size= BATCH_SIZE)

# print(score)

# model.save_weights(savepath1 + 'iv3_1_416.h5')

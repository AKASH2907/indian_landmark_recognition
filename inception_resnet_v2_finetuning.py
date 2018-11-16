import math
import numpy as np
from os.path import isfile, join
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten, Input, AveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import plot_model, np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


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



def build_inception_resnet_V2(img_shape=(512, 512, 3), n_classes=4, l2_reg=0.,
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
    x = Dense(512, activation='swish', name='dense_1', kernel_initializer='he_uniform')(x)
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
x_train = np.load(savepath1 + 'X_train_512.npy')
x_test = np.load(savepath1 +'X_test_512.npy')
x_valid = np.load(savepath1 +'X_valid_512.npy')
y_train = np.load('Y_train.npy')
y_valid = np.load('Y_valid.npy')
y_test = np.load('Y_test.npy')

filepath = savepath1 + "wts/irv2/irv2_512-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose =1, save_best_only=True, mode='max', save_weights_only=True)
# tensorboard = TensorBoard(log_dir=log_path,
#                                 write_graph=False, #This eats a lot of space. Enable with caution!
#                                 #histogram_freq = 1,
#                                 write_images=True,
#                                 batch_size = BATCH_SIZE,
#                                 write_grads=True)
# lrate = LearningRateScheduler(step_decay)

callback = [checkpoint]


model = build_inception_resnet_V2()
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=7, verbose= 1, 
    # steps_per_epoch=x_train.shape[0]//BATCH_SIZE,
    validation_data=(x_valid, y_valid),
    callbacks = callback
    )

# model.save_weights('ir_crops_full.h5')

# model.load_weights(savepath1 + 'wts/irv2/irv2-07-0.80.hdf5')

score = model.evaluate(x_test, y_test, verbose=1, batch_size= BATCH_SIZE)
print(score)

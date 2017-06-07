'''Trains a convnet using ResNet50 pre-trained net. Based partly on code from user fujisan on kaggle.com, accessed at: https://www.kaggle.com/fujisan/use-keras-pre-trained-vgg16-acc-98/versions
Also adapted
from https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
'''

from __future__ import print_function
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import re
import PIL
from PIL import Image
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.utils import np_utils
from keras.applications import ResNet50
from keras import backend as K
import sys
sys.setrecursionlimit(10000)

np.random.seed(1337)  # for reproducibility

# Runs code on GPU
os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

def load_data_from_saved_array():
    data = np.load('validation_set_224.npz')
    x_train = data.files[0]
    x_train = data[x_train]
    x_test = data.files[1]
    x_test = data[x_test]
    y_train = data.files[2]
    y_train = data[y_train]
    y_test = data.files[3]
    y_test = data[y_test]
    return x_train, x_test, y_train, y_test

def cnn_model_resnet50(x_train, x_test, y_train, y_test, batch_size=22, epochs=1, input_shape=(224,224,3)):
    '''
    Builds and runs keras cnn on top of pre-trained ResNet50. Data are generated from X_train.
    '''
    base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(512, activation='relu'))
    # model.add(Activation('relu'))
    # add_model.add(Dropout(0.5))
    add_model.add(Dense(nb_classes, activation='softmax'))
    # model.add(Activation('softmax'))

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    model.summary()

    train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)
    train_datagen.fit(x_train)

    history = model.fit_generator(
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=x_train.shape[0] // batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[ModelCheckpoint('ResNet50.model', monitor='val_acc', save_best_only=False)]
    )
    # model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
    #           verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    ypred = model.predict(x_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return ypred

if __name__ == '__main__':
    nb_classes = 13
    x_train, x_test, y_train, y_test = load_data_from_saved_array()

    ypred = cnn_model_resnet50(x_train, x_test, y_train, y_test, batch_size=1, epochs=1, input_shape=(224,224,3))

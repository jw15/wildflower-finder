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
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
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
import matplotlib.pyplot as plt
sys.setrecursionlimit(10000)

np.random.seed(1337)  # for reproducibility
seed = 1337

# Runs code on GPU
os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"



def train_validation_split(saved_arr ='flowers_224.npz'):
    '''
    Splits train and validation data and images. (Will also load test images, names from saved array).
    Input: saved numpy array, files/columns in that array
    Output: Train/validation data (e.g., X_train, X_test, y_train, y_test), test images, test image names (file names minus '.png')
    '''
    data = np.load('flowers_224.npz')

    x = data.files[0]
    x = data[x]
    y = data.files[1]
    y = data[y]
    # yp = np.array(y)

    # Encode flower categories as numerical
    number = LabelEncoder()
    y = number.fit_transform(y.astype('str'))

    # Split train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1337)
    print('X_train: {} \ny_train: {} \nX_test: {} \ny_test: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

    # Standardize pixel values (between 0 and 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train/255
    X_test = X_test/255

    return X_train, X_test, y_train, y_test

def convert_to_binary_class_matrices(y_train, y_test, nb_classes):
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return Y_train, Y_test



# def load_data_from_saved_array():
#     data = np.load('validation_224.npz')
#     x_train = data.files[0]
#     x_train = data[x_train]
#     x_test = data.files[1]
#     x_test = data[x_test]
#     y_train = data.files[2]
#     y_train = data[y_train]
#     y_test = data.files[3]
#     y_test = data[y_test]
#     return x_train, x_test, y_train, y_test

def cnn_model_resnet50(x_train, x_test, y_train, y_test, batch_size=22, epochs=1, input_shape=(224,224,3)):
    '''
    Builds and runs keras cnn on top of pre-trained ResNet50. Data are generated from X_train.
    '''
    base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(512, activation='relu'))
    # add_model.add(Dropout(0.5))
    add_model.add(Dense(nb_classes, activation='softmax'))

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model.summary()

    train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True)
    seed = 1337
    train_datagen.fit(x_train, seed=seed)

    history = model.fit_generator(
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=(x_train.shape[0] // batch_size),
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[ModelCheckpoint('ResNet50.model', monitor='val_acc', save_best_only=False)]
    )
    # model.fit(x_train, y_train, batch_size=26, epochs=1,
    #           verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    ypred = model.predict(x_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return ypred, model, history

def model_summary_plots(history):
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('../model_plots/model_accuracy_rn50_224x20e_{}.png'.format(seed))
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('../model_plots/model_loss_rn50_224x20e_{}.png'.format(seed))

if __name__ == '__main__':
    nb_classes = 13
    X_train, X_test, y_train, y_test = train_validation_split('flowers_224.npz')

    Y_train, Y_test = convert_to_binary_class_matrices(y_train, y_test, nb_classes)
    # np.savez('val_stratified_224.npz', X_train, X_test, Y_train, Y_test)

    ypred, model, history = cnn_model_resnet50(X_train, X_test, Y_train, Y_test, batch_size=26, epochs=1, input_shape=(224,224,3))
    model_summary_plots(history)
    f = file('../pickles/model_pickle_resnet50_224x20e_{}.pkl'.format(seed), 'wb')
    for obj in [ypred, model, history]:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
'''
save_to_dir='../augmented_images/', save_prefix='aug_', save_format='jpeg',
'''

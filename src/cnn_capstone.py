'''Trains a simple convnet on the initial flower dataset.
Adapted cnn_soln.py, which was
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

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.applications import ResNet50
from keras import backend as K

np.random.seed(1337)  # for reproducibility

# def load_data(directory_path):
    # for filename in os.listdir(directory_path):
        #append filedata to something like a dictionary or mongodb
    #return dictionary or mongodb

# def my_list_dir(root):
#     filelist = os.listdir(root)
#     return [x for x in filelist if not (x.startswith('.'))]

# def clean_file_paths(root):
#     for filename in my_list_dir(root):
#         print(filename)
#         # print(path, subdirs, files)
#         name2 = filename.replace(' ', '')
#         new_name = name2.replace('-', '_')
#         os.replace(filename, new_name)

# def my_image_resize(basewidth, root, resized_root):
#     files = [f for f in listdir(root) if isfile(join(root, f))]
#     resized_files = [f for f in listdir(resized_root) if isfile(join(resized_root, f))]
#     # os.mkdir('../resized_images')
#     # for path, subdirs, files in os.walk(root):
#     for name in files:
#         if not (name.startswith('.')):
#             if not ('{}_resized.png'.format(name[:-4])) in resized_files:
#             # if name != 'cnn_capstone.py':
#                 img = Image.open('{}{}'.format(root, name))
#                 wpercent = (basewidth / float(img.size[0]))
#                 hsize = int((float(img.size[1]) * float(wpercent)))
#                 img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
#                 img.save('../resized_images/{}_resized.png'.format(name[:-4]))

def image_categories(resized_root):
    ''' A dictionary that stores the image path name and flower species for each image
    Input: image path names (from root directory)
    Output: dictionary 'categories'
    '''
    flower_dict = {}
    files = [f for f in listdir(resized_root) if isfile(join(resized_root, f))]
    # for path, subdirs, files in os.walk(resized_root):
        # print(path, subdirs, files)
    for name in files:
        # name = name.replace(' ', '')
        # name = name.replace('-', '_')
        # if not (name.startswith('.')):
        #     if name != 'cnn_capstone.py':
        img_path = '{}/{}'.format(resized_root, name)
        # img_path = os.path.join(path, name)
        img_cat = re.sub("\d+", "", name).rstrip('_resized.png')
        img_cat = img_cat[:-3]
        flower_dict[img_path] = img_cat
    return flower_dict

def my_train_test_split(resized_root):
    X = []
    y = []
    files = [f for f in listdir(resized_root) if isfile(join(resized_root, f))]
    # files = [f for f in os.listdir(os.curdir) if os.path.isfile(f)]
    # resized_files = [f for f in files if f[-11:] == 'resized.png']
    # for path, subdirs, files in os.walk(root):
    for name in files:
        if not (name.startswith('.')):
        #     if name != 'cnn_capstone.py':
                # name = name.replace(' ', '')
                # name = name.replace('-', '_')
            img_path = '{}/{}'.format(resized_root, name)
            # img_path = os.path.join(path, name)
            correct_cat = categories[img_path]
            img_pixels = io.imread('{}/{}'.format(resized_root, name))
            X.append(img_pixels)
            y.append(correct_cat)
    y = np.array(y)
    number = LabelEncoder()
    y = number.fit_transform(y.astype('str'))
    X_arr = np.array(X)
    # X_arr = np.stack(X)
    X_train, X_test, y_train, y_test = train_test_split(X_arr, y)
    return X_train, X_test, y_train, y_test


# def image_reshape(X_train, X_test):
#     if K.image_dim_ordering() == 'th':
#         X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#         X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#         input_shape = (1, img_rows, img_cols)
#     else:
#         X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
#         X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
#         input_shape = (img_rows, img_cols, 1)
#     return X_train, X_test, input_shape

def image_asfloat(X_train, X_test):
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    return X_train, X_test

def image_rgb_unit_scale(X_train, X_test):
    X_train /= 255
    X_test /= 255
    return X_train, X_test

def print_X_shapes(X_train, X_test):
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

def convert_to_binary_class_matrices(y_train, y_test, nb_classes):
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return Y_train, Y_test

def cnn_model(nb_filters, kernel_size, batch_size, nb_epoch, X_test, Y_test, X_train, Y_train):
    model = Sequential()
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                            border_mode='valid',
                            input_shape=(img_rows, img_cols, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    # model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))
    # model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    ypred = model.predict(X_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return ypred

def cnn_model_resnet50(nb_filters, kernel_size, batch_size, nb_epoch, X_test, Y_test, X_train, Y_train, input_shape=(224,224,3)):
    '''
    Builds and runs keras cnn on top of pre-trained ResNet50. Data are generated from X_train.
    '''
    base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(512, activation='relu'))
    # model.add(Activation('relu'))
    add_model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    # model.add(Activation('softmax'))

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    model.summary()


    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    ypred = model.predict(X_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return ypred

if __name__ == '__main__':
    # filelist = my_list_dir('../data_images/')
    # clean_file_paths('../data_images/')
    batch_size = 5
    # number of flowers in dataset
    nb_classes = 11
    #number of repeats of entire model
    nb_epoch = 20

    # input image dimensions
    input_shape = (120, 90
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    root = '../data_images/'
    resized_root = '../resized_images'
    # my_image_resize(90, '../data_images/', resized_root)
    categories = image_categories(resized_root)
    X_train, X_test, y_train, y_test = my_train_test_split(resized_root)
    X_train, X_test = image_asfloat(X_train, X_test)
    X_train, X_test = image_rgb_unit_scale(X_train, X_test)
    print_X_shapes(X_train, X_test)
    Y_train, Y_test = convert_to_binary_class_matrices(y_train, y_test, nb_classes)
    ypred = cnn_model(nb_filters, kernel_size, batch_size, nb_epoch, X_test, Y_test, X_train, Y_train)
    # ypred = cnn_model(nb_filters, kernel_size, batch_size, nb_epoch, X_test, Y_test, X_train, Y_train)

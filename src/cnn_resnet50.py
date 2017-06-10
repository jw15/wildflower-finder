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
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential, model_from_json
from keras.utils import np_utils
from keras.applications import ResNet50
from keras import backend as K
import sys
import matplotlib.pyplot as plt
import pickle

sys.setrecursionlimit(1000000)

seed = 142
np.random.seed(seed)  # for reproducibility

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
    y_labels = np.unique(y)
    flower_cats = {}
    for i, name in enumerate(y_labels):
        flower_cats[i] = name
    y = number.fit_transform(y.astype('str'))

    # Split train and test subsets to get final text data (don't change this)
    X_training, X_test_holdout, y_training, y_test_holdout = train_test_split(x, y, stratify=y, random_state=42, test_size=.2)
    print('Initial split for test data:\n X_training: {} \ny_training: {} \nX_test_holdout: {} \ny_test_holdout: {}'.format(X_training.shape, y_training.shape, X_test_holdout.shape, y_test_holdout.shape))

    # Split train into train and validation data (different for each model):
    X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, random_state=seed, test_size=.2)
    print('Train/validation split for this model:\n X_train: {} \ny_train: {} \nX_test: {} \ny_test: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

    # Standardize pixel values (between 0 and 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_test_holdout = X_test_holdout.astype('float32')
    X_train = X_train/255
    X_test = X_test/255
    X_test_holdout = X_test_holdout/255

    return X_train, X_test, X_test_holdout, y_train, y_test, y_test_holdout, flower_cats

def convert_to_binary_class_matrices(y_train, y_test, y_test_holdout, nb_classes):
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_test_holdout = np_utils.to_categorical(y_test_holdout, nb_classes)
    return Y_train, Y_test, Y_test_holdout

def build_cnn_resnet_50(input_shape=(224,224,3)):
    base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(512, activation='relu'))
    # add_model.add(Dropout(0.5))
    add_model.add(Dense(nb_classes, activation='softmax'))

    # Combine base model and my fully connected layers
    final_model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

    # Specify SGD optimizer parameters
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    # Compile model
    final_model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    return final_model, final_model.summary()

def _image_generator(X_train, Y_train):
    seed = 1337
    train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)
    train_datagen.fit(X_train, seed=seed)
    return train_datagen

    # for batch in ig.flow(X_train, Y_train, seed=seed, batch_size=batch_size):
    #     for i in range(len(batch[0])):
    #         x = batch[0][i].reshape(1,224, 224, 3)
    #         y = batch[i].reshape(1,2)
    #         yield (x, y)

    # train_datagen.fit(x_train, seed=seed)

    # history = model.
    # (
    #     train_datagen.flow(x_train, y_train, batch_size=batch_size),
    #     steps_per_epoch=(x_train.shape[0] // batch_size),
    #     epochs=epochs,
    #     validation_data=(x_test, y_test),
    #     callbacks=callbacks_list

def fit_model_resnet50(X_train, X_test, Y_train, Y_test, batch_size=26, epochs=45, input_shape=(224,224,3)):
    generator = _image_generator(X_train, Y_train)

    # checkpoint
    filepath="weights-improvement142-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

    # Change learning rate when learning plateaus
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
              patience=4, min_lr=0.00001)

    # Stop model once it stops improving to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

    # put all callback functions in a list
    callbacks_list = [checkpoint, reduce_lr]

    history = final_model.fit_generator(
        generator.flow(X_train, Y_train, batch_size=batch_size),
        steps_per_epoch=(X_train.shape[0] // batch_size),
        epochs=epochs,
        validation_data=(X_test, Y_test),
        callbacks=callbacks_list
        )

    # history = final_model.fit_generator(generator,
    # steps_per_epoch=(X_train.shape[0] // batch_size),
    # epochs=epochs,
    # validation_data = (X_test, Y_test),
    # callbacks=callbacks_list
    # )

    score = final_model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
    ypred = final_model.predict(X_test)
    # ypred_classes = final_model.predict_classes(X_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return ypred, final_model, history

# def visualize_layers(model):
#     layer_dict = dict([(layer.name, layer) for layer in model.layers])


'''
def cnn_model_resnet50(x_train, x_test, y_train, y_test, batch_size=22, epochs=50, input_shape=(224,224,3)):
    '''
    #Builds and runs keras cnn on top of pre-trained ResNet50. Data are generated from X_train.
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
    seed = 1337
    train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
            )
    # train_datagen.fit(x_train, seed=seed)

    # Reduce learning rate when model fit plateaus
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
              patience=5, min_lr=0.001)

    # Stop model once it stops improving to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=0, mode='auto')

    # checkpoint
    filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

    # put all callback functions in a list
    callbacks_list = [checkpoint]

    # generate the model, capture model history
    history = model.fit_generator(
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=(x_train.shape[0] // batch_size),
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks_list
        # callbacks=[ModelCheckpoint('ResNet50.model', monitor='val_acc', save_best_only=True), reduce_lr]
    )
    # model.fit(x_train, y_train, batch_size=26, epochs=1,
    #           verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    ypred = model.predict(x_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return ypred, model, history
'''
def model_summary_plots(history):
    print(history.history.keys())
    plt.close('all')
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('../model_plots/model_accuracy_rn50_224x20e_{}.png'.format(seed))
    plt.close('all')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('../model_plots/model_loss_rn50_224_{}.png'.format(seed))

def run_on_test_data(model):
    test_predictions = model.predict(X_test_holdout)
    return test_predictions

if __name__ == '__main__':
    X_train, X_test, X_test_holdout, y_train, y_test, y_test_holdout, flower_cats = train_validation_split('flowers_224.npz')
    nb_classes = len(flower_cats)

    Y_train, Y_test, Y_test_holdout = convert_to_binary_class_matrices(y_train, y_test, y_test_holdout, nb_classes)

    final_model, model_summary = build_cnn_resnet_50(input_shape=(224,224,3))

    ypred, model124, history = fit_model_resnet50(X_train, X_test, Y_train, Y_test, batch_size=26, epochs=40, input_shape=(224,224,3))
    # ypred, model, history = cnn_model_resnet50(X_train, X_test, Y_train, Y_test, batch_size=26, epochs=60, input_shape=(224,224,3))

    # serialize model to JSON
    model_summary_plots(history)
    model_json = model124.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    final_model.save('resnet50_224_124.h5')

    f = open('../model_outputs/history_resnet50_224_{}.pkl'.format(seed), 'wb')
    for obj in [history]:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    np.save('../model_outputs/ypred_rn50_224_{}'.format(seed), ypred, allow_pickle=True)

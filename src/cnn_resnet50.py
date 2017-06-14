'''Trains a convnet using ResNet50 pre-trained net. Based partly on code from user fujisan on kaggle.com, accessed at: https://www.kaggle.com/fujisan/use-keras-pre-trained-vgg16-acc-98/versions
Also adapted
from https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
'''

from __future__ import print_function
import numpy as np
import pandas as pd
import sys, os, re, pickle, datetime, time
from os import listdir
from os.path import isfile, join
from collections import Counter
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras import applications, optimizers, backend as K
from keras.models import Sequential, Model, load_model, model_from_json
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.applications import ResNet50
import matplotlib.pyplot as plt

sys.setrecursionlimit(1000000)

seed = 1337
np.random.seed(seed)  # for reproducibility


# Runs code on GPU
os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

def read_data(data):
    ''' Reads in data loaded from saved numpy array.'''
    x = data.files[0]
    x = data[x]
    y = data.files[1]
    y = data[y]
    return x, y

def class_counts_specifications(y):
    '''
    Makes pandas df for connecting numerical indices to flower categories, counts number of images within each category. Also returns list of class names.
    '''
    class_labels = np.unique(y)
    # flower_cats = {}
    # for i, name in enumerate(class_labels):
    #     flower_cats[i] = name

    flower_cat_counter = Counter(y)

    flower_count_df = pd.DataFrame.from_dict(flower_cat_counter, orient='index')
    flower_count_df = flower_count_df.rename(columns={0: 'species'})
    flower_count_df['count'] = list(flower_cat_counter.values())

    return flower_count_df, class_labels

def train_validation_split(x, y):
    '''
    Splits train and validation data and images. (Will also load test images, names from saved array).
    Input: saved numpy array, files/columns in that array
    Output: Train/validation data (e.g., X_train, X_test, y_train, y_test), test images, test image names (file names minus '.png')
    '''
    # Encode flower categories as numerical values
    number = LabelEncoder()
    y = number.fit_transform(y.astype('str'))

    # Split train and test subsets to get final text data (don't change this)
    X_training, X_test_holdout, y_training, y_test_holdout = train_test_split(x, y, stratify=y, random_state=42, test_size=.2)
    print('Initial split for (holdout) test data:\n \
    X_training: {} \n \
    y_training: {} \n \
    X_test_holdout: {} \n \
    y_test_holdout: {} \n'.format(X_training.shape, y_training.shape, X_test_holdout.shape, y_test_holdout.shape))

    # Split train into train and validation data (different for each model):
    X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, stratify=y_training, random_state=seed, test_size=.2)
    train_classes = len(np.unique(y_train))
    test_classes = len(np.unique(y_test))

    print('Train/validation split for this model:\n \
    X_train: {} \n \
    y_train: {} \n \
    X_test: {} \n \
    y_test: {} \n \
    n_train_classes: {} \n \
    n_test_classes: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape, train_classes, test_classes))

    # Standardize pixel values (between 0 and 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_test_holdout = X_test_holdout.astype('float32')
    X_train = X_train/255
    X_test = X_test/255
    X_test_holdout = X_test_holdout/255

    return X_train, X_test, X_test_holdout, y_train, y_test, y_test_holdout

def convert_to_binary_class_matrices(y_train, y_test, y_test_holdout, nb_classes):
    ''' Converts class vectors to binary class matrices'''
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_test_holdout = np_utils.to_categorical(y_test_holdout, nb_classes)
    return Y_train, Y_test, Y_test_holdout

def build_cnn_resnet_50(input_shape=(224,224,3)):
    ''' Builds and compiles CNN with ResNet50 pre-trained model.
    Input: Shape of images to feed into top layers of model
    Output: Compiled model (final_model), summary of compiled model
    '''
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
    # seed = 135
    train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True)
    train_datagen.fit(X_train, seed)
    return train_datagen

def fit_model_resnet50(X_train, X_test, Y_train, Y_test, save_output_root, model_type, name_time, batch_size, epochs, input_shape):
    print('\nBatch size: {} \nCompiling model...'.format(batch_size))
    generator = _image_generator(X_train, Y_train)

    # checkpoint
    filepath='weights/weights-improvement142-{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

    # Change learning rate when learning plateaus
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
              patience=2, min_lr=0.00001)

    # Stop model once it stops improving to prevent overfitting
    early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=0, mode='auto')

    # put all callback functions in a list
    callbacks_list = [checkpoint, reduce_lr]

    history = final_model.fit_generator(
        generator.flow(X_train, Y_train, batch_size=batch_size),
        steps_per_epoch=(X_train.shape[0] // batch_size),
        epochs=epochs,
        validation_data=(X_test, Y_test),
        callbacks=callbacks_list, shuffle=True
        )

    score = final_model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
    ypred = final_model.predict(X_test)
    # ypred_classes = final_model.predict_classes(X_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return ypred, final_model, history

# def visualize_layers(model):
#     layer_dict = dict([(layer.name, layer) for layer in model.layers])

def model_summary_plots(history, save_output_root, model_type, name_time):
    print(history.history.keys())
    plt.close('all')
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('{}{}_{}/accuracy_plot'.format(save_output_root, model_type, name_time))
    plt.close('all')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig('{}{}_{}/loss_plot'.format(save_output_root, model_type, name_time))

def sklearn_stats(Y_true, y_predicted, target_names):
    predicted_classes = np.argmax(y_predicted, axis=1)
    true_classes = y_test
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    return report

def predictions_from_holdout_data(model_fitted, X_test_holdout, Y_test_holdout):
    test_predictions = model_fitted.predict(X_test_holdout)

    score = model_fitted.evaluate(X_test_holdout, Y_test_holdout, verbose=0, batch_size=batch_size)
    return test_predictions, score

def make_model_summary_file(name_time, save_output_root, model_type, start_time, finish_time, seed, input_shape, epochs, batch_size, nb_classes, X_train, X_test, X_test_holdout, score, flower_count_df, notes):
    with open('{}{}_{}/model_summary.txt'.format(save_output_root, model_type, name_time), 'a') as f:
        f.write('Model Summary \n \
        Start time: {} \n \
        Finish time: {} \n \
        Model: {} \n \
        Seed: {} \n \
        Input shape: {} \n \
        Epochs: {} \n \
        Batch size: {} \n \
        N classes: {} \n \
        X/Y_train size: {} \n \
        X/Y_test size: {} \n \
        X/Y_test_holdout: {} \n \n \
        Test score on X_test: {} \n \
        Accuracy score on X_test: {} \n \
        Flower categories: {} \n \n \
        Notes: {}'.format(start_time, finish_time, model_type, seed, input_shape, epochs, batch_size, nb_classes, len(X_train), len(X_test), len(X_test_holdout), score[0], score[1], flower_count_df, notes))
        # F1 score: {} \n \
        # Precision: {} \n \
        # Recall: {} \n \
        # '

def save_model(name_time, history, model_fitted, flower_count_df, save_output_root, start_time, finish_time, model_type, seed, input_shape, epochs, batch_size, nb_classes, X_train, Y_train, X_test, Y_test, X_test_holdout, Y_test_holdout, score):

    os.mkdir('{}{}_{}'.format(save_output_root, model_type, name_time))
    new_root = '{}{}_{}'.format(save_output_root, model_type, name_time)

    make_model_summary_file(name_time, save_output_root, model_type, start_time, finish_time, seed, input_shape, epochs, batch_size, nb_classes, X_train, X_test, X_test_holdout, score, flower_count_df, notes)

    model_summary_plots(history, save_output_root, model_type, name_time)

    # serialize model to JSON
    model_json = model_fitted.to_json()
    with open('{}{}_{}/model.json'.format(save_output_root, model_type, name_time), "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    with h5py.File('{}{}_{}/{}_{}.h5'.format(save_output_root, model_type, name_time, model_type, name_time), 'w') as f: model_fitted.save('{}{}_{}/{}_{}.h5'.format(save_output_root, model_type, name_time, model_type, name_time))

    # Save model history to pickle
    f = open('{}{}_{}/history.pkl'.format(save_output_root, model_type, name_time), 'wb')
    for obj in [history]:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    # Save df of species indices, image counts per species
    flower_count_df.to_pickle('{}{}_{}/flower_count_df.pkl'.format(save_output_root, model_type, name_time))

    # Save predicted probabilities
    np.save('{}{}_{}/predicted_probas'.format(save_output_root, model_type, name_time), ypred, allow_pickle=True)

    # write_folder_to_bucket(new_root)



if __name__ == '__main__':
    start_time = datetime.datetime.now()
    name_time = time.time()

    # Model descriptors, parameters
    save_output_root = '../model_outputs/'
    model_type = "ResNet50"
    input_shape = (224,224,3)
    epochs = 45
    batch_size = 26
    notes = "SGD; learning rate: .001. Changed steps per epoch from len(x_train)/ batch size to just len(x_train)"

    # Load data from saved numpy array
    data = np.load('flowers_224.npz')
    x, y = read_data(data)

    # Describe data, make pandas df for counts of images and numerical label for species categories
    flower_count_df, class_labels = class_counts_specifications(y)
    nb_classes = len(flower_count_df)

    # Train test validation split
    X_train, X_test, X_test_holdout, y_train, y_test, y_test_holdout = train_validation_split(x, y)
    print('{} classes of flowers'.format(len(flower_count_df)))

    # Convert numerical y to one-hot encoded y
    Y_train, Y_test, Y_test_holdout = convert_to_binary_class_matrices(y_train, y_test, y_test_holdout, nb_classes)

    # Build CNN model
    final_model, model_summary = build_cnn_resnet_50(input_shape)

    # Fit CNN model
    ypred, model_fitted, history = fit_model_resnet50(X_train, X_test, Y_train, Y_test, save_output_root, model_type, name_time, batch_size, epochs, input_shape)

    finish_time = datetime.datetime.now()

    # Get predicted probabilities and evaulate model fit when fitted model run on validation hold out data set
    # test_predictions, score = predictions_from_holdout_data(model_fitted, X_test_holdout, Y_test_holdout)

    # ytest_report = sklearn_stats(y_test, ypred, class_labels)

    test_predictions, score = predictions_from_holdout_data(model_fitted, X_test_holdout, Y_test_holdout)
    # y_holdout_report = sklearn_stats(y_test_holdout, test_predictions, class_labels)

    save_model(name_time, history, model_fitted, flower_count_df, save_output_root, start_time, finish_time, model_type, seed, input_shape, epochs, batch_size, nb_classes, X_train, Y_train, X_test, Y_test, X_test_holdout, Y_test_holdout, score)

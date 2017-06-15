from datetime import datetime
import os, sys, re
from os import listdir
from os.path import isfile, join
from werkzeug import secure_filename
import pickle, theano, pandas as pd, numpy as np
from keras import optimizers
from keras.models import load_model, model_from_json
import os

def image_categories_reverse():
    img_root = 'static/images/img_dict/'
    ''' A dictionary that stores the image path name and flower species for each image
    Input: image path names (from root directory)
    Output: dictionary 'categories'
    '''
    flower_dict = {}
    files = [f for f in listdir(img_root) if isfile(join(img_root, f))]
    for name in files:
        if not (name.startswith('.')):
        #     if name != 'cnn_capstone.py':
            img_path = '{}{}'.format(img_root, name)
            img_cat = re.sub("\d+", "", name).rstrip('_.jpg')
            flower_dict[img_cat] = img_path
    return flower_dict

from datetime import datetime
import os, sys, re
from os import listdir
from os.path import isfile, join
from werkzeug import secure_filename
import pickle, theano, pandas as pd, numpy as np
from keras import optimizers
from keras.models import load_model, model_from_json
from collections import defaultdict
import cv2


def image_categories_reverse():
    img_root = 'static/images/img_dict/'
    ''' A dictionary that stores the image path name and flower species for each image
    Input: image path names (from root directory)
    Output: dictionary 'categories'
    '''
    # flower_dict = {}
    flower_dict = defaultdict(list)
    files = [f for f in listdir(img_root) if isfile(join(img_root, f))]
    for name in files:
        if not (name.startswith('.')):
        #     if name != 'cnn_capstone.py':
            img_path = '{}{}'.format(img_root, name)
            img_cat = re.sub("\d+", "", name).rstrip('_.jpg')
            flower_dict[img_cat].append(img_path)
    return flower_dict

def beautify_name(name):
    name = name.replace("_", " ")
    name = name[0][0].upper() + name[1:]
    return name

def crop_thumbnail(file_path, crop_size):
    '''
    Crops image to new_dims, centering image in frame.
    Input: Image, desired cropped size (crop_size=[height, width])
    Output: Cropped image
    '''
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    row_buffer = (img.shape[0] - crop_size[0]) // 2
    col_buffer = (img.shape[1] - crop_size[1]) // 2
    return img[row_buffer:(img.shape[0] - row_buffer), col_buffer:(img.shape[1] - col_buffer)]

def resize_submission(file_path, crop_size):
    '''
    Crops image to new_dims, centering image in frame.
    Input: Image, desired cropped size (crop_size=[height, width])
    Output: Cropped image
    '''
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    row_buffer = (img.shape[0] - crop_size[0]) // 2
    col_buffer = (img.shape[1] - crop_size[1]) // 2
    return img[row_buffer:(img.shape[0] - row_buffer), col_buffer:(img.shape[1] - col_buffer)]

# def common_names(flower_dict):
#     flower_commons = {}
#     for name in flower_dict.keys():
#         flower_commons[name] =
def make_db():
    cats = ['achillea_lanulosa',
     'adenolinum_lewisii',
     'alyssum_parviflorum',
     'arnica_cordifolia',
     'cerastium_arvense',
     'delphinium_nuttallianum',
     'harbouria_trachypleura',
     'lonicera_tatarica',
     'mertensia_lanceolata',
     'padus_virginiana',
     'penstemon_virens',
     'phlox_longifolia',
     'senecio_integerrimus']

    df = pd.DataFrame(cats)


    # Info from easterncoloradowildflowers.com, coloradowildbuds.com, jeffco.us, southwestcoloradowildflowers.com
    common_names = ['Common Yarrow',
    'Wild Blue Flax',
    'Wild Alyssum, Yellow Alyssum, European Madwort',
    'Heartleaf Arnica',
    'Mouse-ear Chickweed, Alpine Chickweed, Alpine Mouse-ear',
    'Nelson Larkspur',
    'Whiskbroom Parsley',
    'Honeysuckle',
    'Lance-leaf Bluebells',
    'Chokecherry',
    'Bluemist Penstemon',
    'Longleaf Phlox',
    "Lamb's Tongue Ragwort, Common Spring Senecio, Lambstongue Groundsel, Tall Golden Ragwort"]

    df['common_names'] = common_names
    # synonyms: ['',
    # '',
    # 'Alyssum simplex, Alyssum minus',
    #  '',
    #  'Cerastium strictum',
    #  '',
    #  ''
    #  '',
    #  '',
    #  'Prunus virginiana subsp. melanocarpa',
    #  '',
    #  '']
    family = ['Asteraceae (Sunflower)',
    'Linaceae (Flax)',
    'Brassicaceae (Mustard)',
    'Asteraceae (Sunflower)',
    'Caryophyllaceae (Chickweed)',
    'Helleboraceae (Hellebore), Ranunculaceae (Buttercup)',
    'Apiaceae (Parsley)',
    'Caprifoliaceae (Honeysuckle)',
    'Boraginaceae (Forget-Me-Not)',
    'Rosaceae (Rose)',
    'Scrophulariaceae (Figwort)',
    'Polemoniaceae (Phlox)',
    'Asteraceae (Sunflower)']
    df['family'] = family

    veg_zone = ['Foothills, Montane, Subalpine',
    'Plains, Foothills, Montane, Subalpine',
    'Plains, Foothills',
    'Foothills to Subalpine',
    'Montane, Subalpine',
    'Plains, Foothills, Montane',
    'Foothills, Montane (4,400 - 10,000 ft.)',
    'Lower foothills',
    'Foothills to Alpine',
    'Plains/Foothills',
    'Foothills, Montane',
    'Plains, Foothills',
    'Foothills, Montane, Subalpine']
    df['veg_zone'] = veg_zone

    # time_bloom = ['May - Sept',
    # 'Mar - Aug',
    # 'Feb - Mar',
    # 'April - July',
    # 'Spring',
    # 'Mar - July',
    # 'Apr - Jun',
    # 'May - July',
    # 'Apr - Aug'
    # 'Mar - May',
    # 'Mar - Aug',
    # 'Apr - Jun',
    # 'Mar - Jun']
    # df['time_bloom'] = time_bloom
    # habitat: ['',
    # '',
    # 'disturbed areas',
    # 'woodlands',
    # 'meadows',
    # '',
    #
    #
    #
    # 'rocks, gravelly slopes',
    # senecio: 'meadows, wetlands, Oak brush']
    co_status = ['Native',
    'Native',
    'Alien',
    'Native',
    'Native',
    'Native',
    'Native',
    'Introduced',
    'Native',
    'Native',
    'Native',
    'Native',
    'Native']
    df['co_status'] = co_status
    mature_height =['',
    '',
    'to 10 inches',
    '',
    '2-10 inches',
    '',
    '6 - 18 inches',
    '',
    '8 - 14 inches',
    '10 - 15 feet',
    '12 inches',
    '4 - 10 inches',
    'to 2 feet']
    df['mature_height'] = mature_height
    return df

# def make_db():
#     client = MongoClient()
#     database = client['flower_db']
#     coll = database['flower_coll']
#     return coll
#
# def update_db_record():
#     flower_db.flower_colls.update({img_cat: img_cat}, )

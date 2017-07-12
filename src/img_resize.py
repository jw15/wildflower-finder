from __future__ import print_function
import numpy as np
import os, cv2, re, PIL
from os import listdir
from os.path import isfile, join
import re
import PIL
from PIL import Image
from skimage import io
import scipy.misc

def my_image_rename(img_root):
    '''Renames files without spaces in the name"'''
    pathiter = (os.path.join(img_root, name) for root, subdirs, files in os.walk(img_root) for name in files)
    for path in pathiter:
        # newname = path.replace(" ", "")
        # newname = path.replace("-", "_")
        newname = path.replace("_200", "")
        # newname = path.replace(" ", "")
        # newname = path.replace("-", "_")
        # newname = path.replace("arnica_jpg", "sand_lily")
        if newname != path:
            os.rename(path, newname)

def my_image_resize(basewidth, img_root, target_root):
    '''
    Input: desired basewidth for resized images, path for folder containing original images, name for new path for images
    '''
    os.mkdir('{}'.format(target_root))
    files = [f for f in listdir(img_root) if isfile(join(img_root, f))]
    resized_files = [f for f in listdir(target_root) if isfile(join(target_root, f))]
    for name in files:
        if not (name.startswith('.')):
            if not ('{}_resized.png'.format(name[:-4])) in resized_files:
            # if name != 'cnn_capstone.py':
                img = Image.open('{}{}'.format(img_root, name))
                wpercent = (basewidth / float(img.size[0]))
                hsize = int((float(img.size[1]) * float(wpercent)))
                img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
                img.save('{}/{}.jpg'.format(target_root, name[:-4]))

def crop_image(img, crop_size):
    '''
    Crops image to new_dims, centering image in frame.
    Input: Image, desired cropped size (crop_size=[height, width])
    Output: Cropped image
    '''
    row_buffer = (img.shape[0] - crop_size[0]) // 2
    col_buffer = (img.shape[1] - crop_size[1]) // 2
    return img[row_buffer:(img.shape[0] - row_buffer), col_buffer:(img.shape[1] - col_buffer)]

def square_thumbnails(img_root, target_root):
    # my_image_resize(200, img_root, target_root)
    files = [f for f in listdir(img_root) if isfile(join(img_root, f))]
    resized_files = [f for f in listdir(target_root) if isfile(join(target_root, f))]
    for name in files:
        if not (name.startswith('.')):
            if not ('{}_resized.png'.format(name[:-4])) in resized_files:
                img_full_path = '../capstone_web_app/static/images/img_dict/{}'.format(name)
                img = cv2.imread(img_full_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = crop_image(img, (200, 200))
                scipy.misc.toimage(img).save('{}/{}'.format(target_root, name))

                # cv2.imwrite('{}/{}'.format(target_root, name), img)
                # img.save('{}/{}.jpg'.format(target_root, name[:-4]))
def resize_thumbnails(basewidth, img_root, target_root, crop_size):
    files = [f for f in listdir(img_root) if isfile(join(img_root, f))]
    resized_files = [f for f in listdir(target_root) if isfile(join(target_root, f))]
    for name in files:
        if not (name.startswith('.')):
            if not ('{}_resized.png'.format(name[:-4])) in resized_files:
            # if name != 'cnn_capstone.py':
                img_full_path = '../capstone_web_app/static/images/img_dict/{}'.format(name)
                img = cv2.imread(img_full_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = Image.open('{}{}'.format(img_root, name))
                wpercent = (basewidth / float(img.shape[0]))
                hsize = int((float(img.shape[1]) * float(wpercent)))
                img = cv2.resize(img, (200, 267))
                img = crop_image(img, crop_size)
                scipy.misc.toimage(img, cmin=0.0, cmax=...).save('{}/{}'.format(target_root, name))

if __name__ == '__main__':
    img_root = '../capstone_web_app/static/images/img_dict/'
    target_root = '../capstone_web_app/static/images/thumbs_200w'
    os.mkdir('{}'.format(target_root))
    # my_image_rename(img_root)
    resize_thumbnails(200, img_root, target_root, (200, 200))
    # my_image_resize(200, img_root, target_root)
    # square_thumbnails(200, img_root, target_root, (200, 200))

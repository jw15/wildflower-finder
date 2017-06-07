from __future__ import print_function
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import re
import PIL
from PIL import Image
from skimage import io

def my_image_rename(img_root):
    '''Renames files without spaces in the name"'''
    pathiter = (os.path.join(img_root, name) for root, subdirs, files in os.walk(img_root) for name in files)
    for path in pathiter:
        newname = path.replace(" ", "")
        newname = path.replace("-", "_")
        # newname = path.replace(" ", "")
        # newname = path.replace("-", "_")
        # newname = path.replace("arnica_jpg", "sand_lily")
        if newname != path:
            os.rename(path, newname)

def my_image_resize(basewidth, img_root, target_root):
    '''
    Input: desired basewidth for resized images, path for folder containing original images, name for new path for images
    '''
    files = [f for f in listdir(img_root) if isfile(join(img_root, f))]
    os.mkdir('{}'.format(target_root))
    resized_files = [f for f in listdir(target_root) if isfile(join(target_root, f))]
    for name in files:
        if not (name.startswith('.')):
            if not ('{}_resized.png'.format(name[:-4])) in resized_files:
            # if name != 'cnn_capstone.py':
                img = Image.open('{}{}'.format(img_root, name))
                wpercent = (basewidth / float(img.size[0]))
                hsize = int((float(img.size[1]) * float(wpercent)))
                img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
                img.save('{}/{}_{}.png'.format(target_root, name[:-4], basewidth))


if __name__ == '__main__':
    img_root = '../imgs_for_readme/'
    target_root = '../imgs_for_readme_rsz'
    # my_image_rename(img_root)
    my_image_resize(120, img_root, target_root)

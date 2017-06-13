'''
Image preprocessing for CNN.
Input: jpg files exported from Mac Photo app
Output: Saved numpy array of images in specified shape for use in CNN. Corrects for class imbalance via image generation.
Images are resized to specified shape, cropped to square, then image generation is used to create flipped/rotated images.
'''

from __future__ import print_function
import numpy as np
import pandas as pd
import os, cv2, re, PIL
from os import listdir
from os.path import isfile, join
from PIL import Image
from skimage import io
from skimage.transform import resize

from img_resize import my_image_resize
from img_resize import my_image_rename

np.random.seed(1337)  # for reproducibility


def _center_image(img, new_size=[256, 256]):
    '''
    Helper function. Takes rectangular image resized to be max length on at least one side and centers it in a black square.
    Input: Image (usually rectangular - if square, this function is not needed).
    Output: Image, centered in square of given size with black empty space (if rectangular).
    '''
    row_buffer = (new_size[0] - img.shape[0]) // 2
    col_buffer = (new_size[1] - img.shape[1]) // 2
    centered = np.zeros(new_size + [img.shape[2]], dtype=np.uint8)
    centered[row_buffer:(row_buffer + img.shape[0]), col_buffer:(col_buffer + img.shape[1])] = img
    return centered

def resize_image_to_square(img, new_size=((256, 256))):
    '''
    Resizes images without changing aspect ratio. Centers image in square black box.
    Input: Image, desired new size (new_size = [height, width]))
    Output: Resized image, centered in black box with dimensions new_size
    '''
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*new_size[1]/img.shape[0]),new_size[1])
    else:
        tile_size = (new_size[1], int(img.shape[0]*new_size[1]/img.shape[1]))
    # print(cv2.resize(img, dsize=tile_size))
    return _center_image(cv2.resize(img, dsize=tile_size), new_size)

def crop_image(img, crop_size):
    '''
    Crops image to new_dims, centering image in frame.
    Input: Image, desired cropped size (crop_size=[height, width])
    Output: Cropped image
    '''
    row_buffer = (img.shape[0] - crop_size[0]) // 2
    col_buffer = (img.shape[1] - crop_size[1]) // 2
    return img[row_buffer:(img.shape[0] - row_buffer), col_buffer:(img.shape[1] - col_buffer)]

def process_image(image_path, resize_new_size=[256,256], crop_size=[224, 224]):
    '''
    Input: File name (image)
    Output: numpy array of processed image: normalized, resized, centered)
    '''
    x = []
    # capstone_web_app/predicted_images_web/cinquefoil_extra.jpg
    img_full_path = 'predicted_images_web/{}'.format(image_path)
    # print(img_full_path)
    img = cv2.imread(img_full_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_image_to_square(img, resize_new_size)
    img = crop_image(img, crop_size)
    x = np.array(img)
    x = x.reshape((1,) + x.shape)
    return x


# if __name__ == '__main__':
#     x = process_image('cinquefoil_extra.jpg')
#     image_array = process_image(file_list, resize_new_size=[256,256], crop_size=[224, 224])
#     np.savez('flowers_224_2.npz', image_array, y)

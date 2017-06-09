from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

data = np.load('flowers_224.npz')

x = data.files[0]
x = data[x]
y = data.files[1]
y = data[y]

seed = 1337
datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

img = x[610]
img = img.reshape((1,) + img.shape)

i = 0
for batch in datagen.flow(img, batch_size=1,
                      save_to_dir='preview', save_prefix='aug', save_format='jpeg'):
    i += 1
    if i > 10:
        break

# if __name__ == '__main__':
#     datagen = _image_generator(X_)

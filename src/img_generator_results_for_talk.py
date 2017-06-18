from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import PIL

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
for batch in datagen.flow(img, batch_size=1, save_to_dir='preview', save_prefix='aug', save_format='jpeg'):
    i += 1
    if i > 10:
        break

# save images to file for use in imagemagic montage
def save_x_to_pngs(x, target_root):
    img_dict = {}
    for i in range(len(x)):
        img = PIL.Image.fromarray(x[i])
        img_path = '{}/img{}.png'.format(target_root, i)
        img.save(img_path)
        img_dict[i] = img_path
    return img_dict




# if __name__ == '__main__':
#     datagen = _image_generator(X_)

Making a photo montage using ImageMagick

from http://www.imagemagick.org/Usage/montage/

Put all the image files in one folder.
Save the image file names to a txt file (in terminal):
ls > contents.txt

Go into text file, delete name of text file, delete all \n's.

# In Python:
def list_new_files(img_dict):
    # img_root = '../saved_x'
    # files = [f for f in listdir(img_root) if isfile(join(img_root, f))]
    # clean = open('imgs_square.txt').read().replace('\n', ' ')
    # clean = clean.split(' ')
    file_list = list(img_dict.values())
    indexes = list(img_dict.keys())
    achillea_files = []
    for i in range(0, 118):
        file_add = img_dict[i]
        achillea_files.append(file_add)
    achillea_files = ' '.join(achillea_files)
    with open("txt_files/achillea_clean.txt", "w") as text_file:
        print((achillea_files), file=text_file)

    adenolinum_files = []
    for i in range(118, 302):
        file_add = img_dict[i]
        adenolinum_files.append(file_add)
    adenolinum_files = ' '.join(adenolinum_files)
    with open("txt_files/adenolinum_clean.txt", "w") as text_file:
        print((adenolinum_files), file=text_file)


    


In bash terminal:

montage <file names (copy/paste)> \ -tile 5x2 \ -geometry 48x48\>+1+1 \ aug_photos.jpg




montage aug_0_1752.jpeg aug_0_2515.jpeg aug_0_4082.jpeg aug_0_4310.jpeg aug_0_4755.jpeg aug_0_481.jpeg aug_0_5629.jpeg aug_0_5845.jpeg aug_0_721.jpeg aug_0_7456.jpeg \ -tile 5x2 \ -geometry 48x48\>+1+1 \ aug_photos.jpg

# redo with different images:

montage aug_0_1193.jpeg aug_0_1921.jpeg aug_0_2547.jpeg aug_0_3236.jpeg aug_0_4920.jpeg aug_0_4924.jpeg aug_0_5426.jpeg aug_0_5877.jpeg aug_0_5972.jpeg aug_0_7877.jpeg aug_0_8810.jpeg \-tile 2x5 \-geometry 120x120\>+1+1 \aug_photos_flax_224_4.jpg


montage img0.png img1.png img2.png img3.png img4.png img5.png img6.png img7.png img8.png img9.png img10.png  \-tile 1x10 \-geometry 25x25\> \achillea_20_15.jpg


montage img118.png img119.png img120.png img121.png img122.png img123.png img124.png img125.png img126.png img127.png img128.png img129.png \-tile 1x10 \-geometry 25x25\> \adenolinum_20_15.jpg


if __name__ == '__main__':
    img_dict = save_x_to_pngs(x, target_root)

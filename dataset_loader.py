import h5py
import glob
import os
import numpy as np
import cv2
from sklearn.utils import shuffle
from keras.utils import to_categorical

IM_WIDTH = 224
IM_HEIGHT = 224
SRC_WIDTH = 400
SRC_HEIGHT = 273

NEW_WIDTH = int(IM_HEIGHT / SRC_HEIGHT * SRC_WIDTH)
CROP_LIMIT = (NEW_WIDTH - IM_WIDTH) // 2


def load_image(path: str):
    mean_pixel = [103.939, 116.779, 123.68]
    im = cv2.imread(path).astype(np.float32)
    im = cv2.resize(im, (NEW_WIDTH, IM_HEIGHT))  # resize image to get the VGG expected size (adapt VGG ?)
    im = im[:, CROP_LIMIT:CROP_LIMIT + IM_WIDTH, :]  # crop a square in the center of the image...
    for c in range(3):
        im[:, :, c] -= mean_pixel[c]
    # im = im.transpose((2, 0, 1))
    return im


def load_image_pair(path: str):
    im1 = load_image(path + "image1.png")
    im2 = load_image(path + "image2.png")
    return im1, im2


def load_data_set():
    try:
        with h5py.File('data/X.h5') as hf:
            x_1, x_2, y = hf['img1s'][:], hf['img2s'][:], hf['labels'][:]
        print("Loaded images from X.h5")
        return x_1, x_2, y
    except (IOError, OSError, KeyError):
        print("Error in reading X.h5. Processing all images...")
        root_dir = 'data/'
        img1s = []
        img2s = []
        labels = []

        sim_img_paths = glob.glob(os.path.join(root_dir, '[12]/*/'))
        dif_img_paths = glob.glob(os.path.join(root_dir, '0/*/'))
        np.random.shuffle(sim_img_paths)
        np.random.shuffle(dif_img_paths)
        for img_path in sim_img_paths:
            try:
                im1, im2 = load_image_pair(img_path)
                label = 1
                img1s.append(im1)
                img2s.append(im2)
                labels.append(label)

                if len(img1s) % 100 == 0:
                    print("Processed {}/{}".format(len(img1s), len(sim_img_paths) + len(dif_img_paths)))
            except (IOError, OSError, AttributeError):
                print('missed', img_path)
                pass
        for img_path in dif_img_paths:
            try:
                im1, im2 = load_image_pair(img_path)
                label = 0
                img1s.append(im1)
                img2s.append(im2)
                labels.append(label)

                if len(img1s) % 100 == 0:
                    print("Processed {}/{}".format(len(img1s), len(sim_img_paths) + len(dif_img_paths)))
            except (IOError, OSError, AttributeError):
                print('missed', img_path)
                pass
        print("Processed {}/{}".format(len(img1s), len(sim_img_paths) + len(dif_img_paths)))

        x_1 = np.array(img1s, dtype='float32')
        x_2 = np.array(img2s, dtype='float32')
        y = np.array(to_categorical(labels), dtype=np.int)

        x_1, x_2, y = shuffle(x_1, x_2, y)

        with h5py.File('data/X.h5', 'w') as hf:
            hf.create_dataset('img1s', data=x_1)
            hf.create_dataset('img2s', data=x_2)
            hf.create_dataset('labels', data=y)

        return x_1, x_2, y

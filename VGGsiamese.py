from keras.applications import VGG16
from keras.layers import Dense, Input
from keras.models import Model
import h5py
import glob
import os
import numpy as np
import cv2

from siamese import get_siamese_model


def get_siamese_vgg_model(image_shape=(224, 224, 3)):
    input_a = Input(image_shape)
    input_b = Input(image_shape)

    vgg_base = VGG16(include_top=False, input_shape=image_shape, weights='imagenet')

    siamese_vgg_model = get_siamese_model(vgg_base, image_shape)

    top = Dense(128, activation="relu")(siamese_vgg_model([input_a, input_b]))
    top = Dense(1, activation="sigmoid")(top)

    return Model([input_a, input_b], top)


def load_image(path: str):
    mean_pixel = [103.939, 116.779, 123.68]
    im = cv2.imread(path).astype(np.float32)
    for c in range(3):
        im[:, :, c] -= mean_pixel[c]
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return im


def load_image_pair(path: str):
    im1 = load_image(path + "image1.png")
    im2 = load_image(path + "image2.png")
    return im1, im2


def load_data_set():
    try:
        with h5py.File('X.h5') as hf:
            X, Y = hf['imgs'][:], hf['labels'][:]
        print("Loaded images from X.h5")

    except (IOError, OSError, KeyError):
        print("Error in reading X.h5. Processing all images...")
        root_dir = 'GTSRB/Final_Training/Images/'
        imgs = []
        labels = []

        sim_img_paths = glob.glob(os.path.join(root_dir, '1/*/'))
        dif_img_paths = glob.glob(os.path.join(root_dir, '-1/*/'))
        np.random.shuffle(sim_img_paths)
        np.random.shuffle(dif_img_paths)
        for img_path in sim_img_paths:
            try:
                im1, im2 = load_image_pair(img_path)
                label = True
                imgs.append((im1, im2))
                labels.append(label)

                if len(imgs) % 100 == 0:
                    print("Processed {}/{}".format(len(imgs), len(sim_img_paths) + len(dif_img_paths)))
            except (IOError, OSError):
                print('missed', img_path)
                pass
        for img_path in dif_img_paths:
            try:
                im1, im2 = load_image_pair(img_path)
                label = False
                imgs.append((im1, im2))
                labels.append(label)

                if len(imgs) % 100 == 0:
                    print("Processed {}/{}".format(len(imgs), len(sim_img_paths) + len(dif_img_paths)))
            except (IOError, OSError):
                print('missed', img_path)
                pass

        X = np.array(imgs, dtype='float32')
        Y = np.eye(2, dtype='uint8')[labels]

        with h5py.File('X.h5', 'w') as hf:
            hf.create_dataset('imgs', data=X)
            hf.create_dataset('labels', data=Y)


if __name__ == '__main__':
    model = get_siamese_vgg_model()

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])

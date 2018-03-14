from keras.applications import VGG16
from keras.layers import Dense, Input
from keras.models import Model
import h5py
import glob
import os

from siamese import get_siamese_model


def get_siamese_vgg_model(image_shape=(224, 224, 3)):
    input_a = Input(image_shape)
    input_b = Input(image_shape)

    vgg_base = VGG16(include_top=False, input_shape=image_shape, weights='imagenet')

    siamese_vgg_model = get_siamese_model(vgg_base, image_shape)

    top = Dense(128, activation="relu")(siamese_vgg_model([input_a, input_b]))
    top = Dense(1, activation="sigmoid")(top)

    return Model([input_a, input_b], top)


def load_data_set(path: str):
    try:
        with h5py.File('X.h5') as hf:
            X, Y = hf['imgs'][:], hf['labels'][:]
        print("Loaded images from X.h5")

    except (IOError, OSError, KeyError):
        print("Error in reading X.h5. Processing all images...")
        root_dir = 'GTSRB/Final_Training/Images/'
        imgs = []
        labels = []

        all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
        np.random.shuffle(all_img_paths)
        for img_path in all_img_paths:
            try:
                img = preprocess_img(io.imread(img_path))
                label = get_class(img_path)
                imgs.append(img)
                labels.append(label)

                if len(imgs) % 1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
            except (IOError, OSError):
                print('missed', img_path)
                pass

        X = np.array(imgs, dtype='float32')
        Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

        with h5py.File('X.h5', 'w') as hf:
            hf.create_dataset('imgs', data=X)
            hf.create_dataset('labels', data=Y)


if __name__ == '__main__':
    model = get_siamese_vgg_model()

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])

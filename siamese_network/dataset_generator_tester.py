import glob
import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from siamese_network.siamese import data_triple_generator_from_dir


def main():
    dataset_path = "data"
    batch_size = 100
    out_dir = "generated_pair_examples"
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=5,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=[0.6, 1.1],
        fill_mode='reflect',
        horizontal_flip=True,
        vertical_flip=False,
        # validation_split=1.0 - train_ratio,
        brightness_range=(0.7, 1.3)
    )
    train_images_paths = glob.glob(os.path.join("data", 'train/*/*/*'))
    datagen.fit(np.array(list(map(cv2.imread, train_images_paths[:200]))))
    # init triple generator (image 1, image 2, label)
    triple_generator = data_triple_generator_from_dir(datagen,
                                                      os.path.join(dataset_path, 'train'),
                                                      batch_size,
                                                      include_label=True)
    os.makedirs(out_dir, exist_ok=True)
    for i, ((im1, im2), label) in enumerate(triple_generator):
        if i > batch_size:
            break
        print(i, end='\r')
        print(im1.shape, im2.shape)
        cv2.imwrite(os.path.join(out_dir, "{}_{:04d}_left.png".format(label, i)), im1)
        cv2.imwrite(os.path.join(out_dir, "{}_{:04d}_right.png".format(label, i)), im2)


if __name__ == '__main__':
    main()

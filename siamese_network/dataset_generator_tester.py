import glob
import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from siamese_network.siamese import data_triple_generator_from_dir


def main():
    dataset_path = "data"
    batch_size = 100
    number_of_saved_batch = 1
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
    datagen.fit(np.array(list(map(cv2.imread, train_images_paths[:20]))))
    # init triple generator (image 1, image 2, label)
    triple_generator = data_triple_generator_from_dir(datagen,
                                                      os.path.join(dataset_path, 'train'),
                                                      batch_size,
                                                      include_label=True)
    os.makedirs(out_dir, exist_ok=True)
    for b, ((im_batch_1, im_batch_2), label) in enumerate(triple_generator):
        for i, (im1, im2) in enumerate(zip(im_batch_1, im_batch_2)):
            path = os.path.join(out_dir, "{}_{:04d}_left.png".format(label[i][0], i))
            print(i + b * batch_size, path, end='\r')
            cv2.imwrite(path,
                        cv2.cvtColor((im1 + 2.0) * 64, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(out_dir, "{}_{:04d}_right.png".format(label[i][0], i)),
                        cv2.cvtColor((im2 + 2.0) * 64, cv2.COLOR_RGB2BGR))
        if b >= number_of_saved_batch:
            break
        print("done")


if __name__ == '__main__':
    main()

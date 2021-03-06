import os
import glob
import cv2
import numpy as np
import csv
import socket
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import load_model

from siamese_network.siamese import data_triple_generator_from_dir

if socket.gethostname() == "k20c-3":
    base_path = "/mnt/lake/VBags"
else:
    base_path = "/cs-share/pradalier/lake/VBags"


def get_gps_coord(d, seq):
    with open(os.path.join(base_path, d, "image_auxilliary.csv"), mode='r') as d_file:
        reader = csv.reader(d_file)
        for line in reader:
            if line[1] == seq + ".000000":
                return line[2], line[3]
        raise ValueError("{} not fund in {}".format(seq, d))


class DatasetTester:
    def __init__(self, dataset_path="./data2", datagen_test: ImageDataGenerator=None):
        self.dataset_base_path = dataset_path
        if datagen_test is None:
            self.datagen = ImageDataGenerator(
                    featurewise_center=True,
                    featurewise_std_normalization=True,
                    rotation_range=0,
                    width_shift_range=0.0,
                    height_shift_range=0.0,
                    zoom_range=[1.0, 1.0],
                    fill_mode='reflect',
                    horizontal_flip=False,
                    vertical_flip=False,
                    brightness_range=[1.0, 1.0]
                )
            # fit data generator on the firsts 90 images of the dataset
            self.datagen.fit(np.array(list(map(cv2.imread,
                                               glob.glob(os.path.join(dataset_path, 'train/*/*/*.jpg'))[:90]))))
        else:
            self.datagen = datagen_test

    def evaluate(self, model: Model, mode: str="train", batch_size: int=32, add_coordinate: bool=True):
        if mode == "train":
            dataset = 'train'
        elif mode == "test":
            dataset = 'test'
        elif mode == "localisation":
            dataset = 'localisation'
        elif mode == "time":
            dataset = 'time'
        else:
            train_result = self.evaluate(model, mode='train', batch_size=batch_size, add_coordinate=add_coordinate)
            test_result = self.evaluate(model, mode='test', batch_size=batch_size, add_coordinate=add_coordinate)
            return train_result + test_result

        dataset_path_left = os.path.join(self.dataset_base_path, dataset, "left/*/*")
        dataset_path_right = os.path.join(self.dataset_base_path, dataset, "right/*/*")
        file_list_left = glob.glob(dataset_path_left)
        file_list_right = glob.glob(dataset_path_right)
        number_of_file = len(file_list_left)
        if number_of_file != len(file_list_right):
            raise ValueError("{}: {} and {} are not the same".format(mode, number_of_file, len(file_list_right)))

        triple_generator = data_triple_generator_from_dir(datagen=self.datagen,
                                                          dataset_dir=os.path.join(self.dataset_base_path, dataset),
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          include_label=False)

        prediction = model.predict_generator(generator=triple_generator,
                                             steps=number_of_file // batch_size + 1)

        result_list = []

        for file_left, file_right, p in zip(file_list_left, file_list_right, prediction[:number_of_file]):
            label = file_left.split('/')[-2]
            if label != file_right.split('/')[-2]:
                print("Warning:", file_left, "and", file_right, "are probably not a pair")
            if label == '1':
                score = p[0]
            else:
                score = p[1]
            if not add_coordinate:
                result_list.append((file_left, file_right, score))
            else:
                file_name = os.path.split(file_left)[-1]
                file_name = file_name[:-4]  # get read of file extension (assuming that it's .png or .jpg)
                file_descriptor = file_name.split('_')
                if len(file_descriptor) == 8:
                    x, y = file_descriptor[2:4]
                elif len(file_descriptor) == 2:
                    d, seq = file_descriptor[0].split('-')
                    try:
                        x, y = get_gps_coord(d, seq)
                    except ValueError as e:
                        print("Error when loading coordinate:", str(e))
                        continue
                else:
                    print("Incorrect file name:", file_name)
                    continue
                result_list.append((file_left, file_right, score, x, y))

        return result_list









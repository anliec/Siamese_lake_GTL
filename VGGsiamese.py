from keras.applications import VGG16
from keras.layers import Dense, Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
import glob
import os
import sys
import cv2
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import random

from dataset_loader import load_data_set
from siamese import get_siamese_model


K.set_image_dim_ordering('tf')


def get_siamese_vgg_model(image_shape=(224, 224, 3), weights='imagenet', train_from_layers: int=19,
                          merge_type='concatenate', add_batch_norm=False):
    input_a = Input(image_shape)
    input_b = Input(image_shape)

    vgg_base = VGG16(include_top=False,
                     input_shape=image_shape,
                     weights=weights)

    for layer in vgg_base.layers[:train_from_layers]:
        layer.trainable = False

    # vgg_base.summary()

    siamese_vgg_model = get_siamese_model(vgg_base, image_shape,
                                          add_batch_norm=add_batch_norm,
                                          merge_type=merge_type)

    siamese_vgg_model.summary()

    top = Dense(128, activation="relu")(siamese_vgg_model([input_a, input_b]))
    # top = Dense(128, activation="relu")(top)
    top = Dense(2, activation="sigmoid")(top)

    return Model(inputs=[input_a, input_b], outputs=top)


def data_triple_generator(datagen: ImageDataGenerator, x_im1: np.ndarray, x_im2: np.ndarray, y: np.ndarray, batch_size: int):
    for i, ((im1, label), (im2, _)) in enumerate(zip(datagen.flow(x_im1, y, batch_size=batch_size),
                                                     datagen.flow(x_im2, y, batch_size=batch_size))):
        if random.random() <= 0.5:
            yield [im1, im2], label
        else:
            yield [im2, im1], label
        # add vertical flip on both images ? randomly switch images ?


def data_triple_generator_from_dir(datagen: ImageDataGenerator, dataset_dir, batch_size: int, seed=6,
                                   save_to_dir: str=None, shuffle=True):
    left_sav_dir, right_sav_dir = None, None
    if save_to_dir is not None:
        left_sav_dir += '/left'
        right_sav_dir += '/right'
    left_flow = datagen.flow_from_directory(directory=dataset_dir + '/left',
                                            class_mode='categorical',
                                            batch_size=batch_size,
                                            target_size=(224, 224),
                                            shuffle=shuffle,
                                            seed=seed,
                                            save_to_dir=left_sav_dir)
    right_flow = datagen.flow_from_directory(directory=dataset_dir + '/right',
                                             class_mode='categorical',
                                             batch_size=batch_size,
                                             target_size=(224, 224),
                                             shuffle=shuffle,
                                             seed=seed,
                                             save_to_dir=right_sav_dir)
    for (im1, label1), (im2, label2) in zip(left_flow, right_flow):
        # assert (label1 == label2).all()
        if random.random() <= 0.5:
            yield [im1, im2], label1
        else:
            yield [im2, im1], label1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size',
                        default=40,
                        type=int,
                        dest="batch_size")
    parser.add_argument('-bn', '--batch-norm',
                        default=False,
                        type=bool,
                        dest="use_batch_norm")
    parser.add_argument('-m', '--merge-layer',
                        default='concatenate',
                        type=str,
                        dest="merge_layer")
    parser.add_argument('-vl', '--vgg-frozen-limit',
                        default=19,
                        type=int,
                        dest="vgg_frozen_layer")
    parser.add_argument('-o', '--out-dir',
                        default=None,
                        type=str,
                        dest="output_dir")
    parser.add_argument('-e', '--epochs',
                        default=1,
                        type=int,
                        dest="number_of_epoch")
    parser.add_argument('-f', '--fine-tuning-iteration',
                        default=0,
                        type=int,
                        dest="fine_tuning_iteration")
    args = parser.parse_args()
    # batch_size = 1647
    # batch_size = 40
    # train_ratio = 0.9

    # x_1, x_2, y = load_data_set()
    # train_size = int(train_ratio * len(y))
    # x_1_train, x_2_train, y_train = x_1[:train_size], x_2[:train_size], y[:train_size]
    # x_1_test, x_2_test, y_test = x_1[train_size:], x_2[train_size:], y[train_size:]
    
    model = get_siamese_vgg_model(add_batch_norm=args.use_batch_norm,
                                  train_from_layers=args.vgg_frozen_layer,
                                  merge_type=args.merge_layer)
    # model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

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
    datagen_test = ImageDataGenerator(
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

    # img_paths = glob.glob(os.path.join("data2/test/", '*/[01]/*.png'))
    # fit_images = np.array(map(cv2.imread, img_paths))
    datagen.fit(np.array(list(map(cv2.imread, glob.glob(os.path.join("data2/test/", '*/[01]/*.png'))))))

    print("fit done")

    # triple_generator = data_triple_generator(datagen, x_1_train, x_2_train, y_train, batch_size)
    triple_generator = data_triple_generator_from_dir(datagen, "data2/train", args.batch_size)
    triple_generator_test = data_triple_generator_from_dir(datagen_test, "data2/test", args.batch_size, shuffle=False)
    
    df_history = pd.DataFrame()
    for fine_tun_iteration in range(args.fine_tuning_iteration + 1):
        history = model.fit_generator(generator=triple_generator,
                                      steps_per_epoch=1201 // args.batch_size + 1,
                                      epochs=args.number_of_epoch // fine_tun_iteration + 1,
                                      verbose=1,
                                      validation_data=triple_generator_test,
                                      validation_steps=160 // args.batch_size + 1,
                                      initial_epoch=args.number_of_epoch * fine_tun_iteration
                                      )
        epoch = history.epoch
        h_values = history.history.values()
        values = np.array([epoch, ] + list(h_values))
        df = pd.DataFrame(data=values.T, columns=["epoch", ] + list(history.history.keys()))
        if df_history.shape == (0, 0):
            df_history = df
        else:
            df_history = df_history.append(df)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        # write history to csv for later use

        df_history.to_csv(os.path.join(args.output_dir, 'history.csv'),
                          sep=',')
        # do some fancy plots
        # Accuracy
        plt.figure()
        plt.plot(df_history.get('val_categorical_accuracy'), df_history.get('epoch'), label='val_categorical_accuracy')
        plt.plot(df_history.get('categorical_accuracy', df_history.get('epoch')), label='categorical_accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig(os.path.join(args.output_dir, 'accuracy.png'))
        # Loss
        plt.figure()
        plt.plot(df_history.get('val_loss'), df_history.get('epoch'), label='val_loss')
        plt.plot(df_history.get('loss'), df_history.get('epoch'), label='loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig(os.path.join(args.output_dir, 'loss.png'))

        with open(os.path.join(args.output_dir, 'args.txt'), mode='w') as f:
            f.write(str(sys.argv))


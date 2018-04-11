import matplotlib
matplotlib.use('Agg')

from keras.applications import VGG16, ResNet50
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
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
from siamese import get_siamese_layers

TRAIN_DATASET_SIZE = 10706
TEST_DATASET_SIZE = 1299


K.set_image_dim_ordering('tf')


def get_siamese_vgg_model(image_shape=(224, 224, 3), weights='imagenet', train_from_layers: int=19,
                          merge_type='concatenate', add_batch_norm=False, layer_block_to_remove=0,
                          dropout=None):
    """
    create a VGG based, siamese network
    :param image_shape: shape of input ((224, 224, 3) if the original weight are used)
    :param weights: source of the weight to use on VGG, can by 'imagenet' or 'None' (or a path to weights)
    :param train_from_layers: number of layers to froze starting from the first layer (default 19: all)
    :param merge_type: how to merge the output of the siamese network, on of 'dot', 'multiply',
        'subtract', 'l1', 'l2' or 'concatenate'.
    :param add_batch_norm: True to add batch normalization before merging layers (default: False)
    :param layer_block_to_remove: How many block of layers to remove from VGG, starting at the end
        can be 0, 1, 2, 3 or 4, default 0.
    :param dropout: float: dropout to add in top model, None: no dropout (default: None)
    :return: a Keras model of a siamese VGG model with the given parameters
    """
    if 0 > layer_block_to_remove > 4:
        raise ValueError("layer_block_to_remove only can be 0, 1, 2, 3 or 4")

    input_a = Input(image_shape)
    input_b = Input(image_shape)

    vgg_base = VGG16(include_top=False,
                     input_shape=image_shape,
                     weights=weights)

    for layer in vgg_base.layers[:train_from_layers]:
        layer.trainable = False

    for i in range(layer_block_to_remove):
        vgg_base.layers.pop()  # max poolling
        if i < 3:
            vgg_base.layers.pop()  # conv 3
        vgg_base.layers.pop()  # conv 2
        vgg_base.layers.pop()  # conv 1

    vgg_base = Model(vgg_base.input, vgg_base.layers[-1].output)
    vgg_base.summary()

    siamese_vgg = get_siamese_layers(vgg_base, input_a, input_b,
                                     add_batch_norm=add_batch_norm,
                                     merge_type=merge_type)

    # top = Dense(512, activation="relu")(siamese_vgg)
    # if dropout is not None:
    #     top = Dropout(dropout)(top)
    top = Dense(128, activation="relu")(siamese_vgg)
    if dropout is not None:
        top = Dropout(dropout)(top)
    top = Dense(2, activation="softmax")(top)

    return Model(inputs=[input_a, input_b], outputs=top)


def unfroze_core_model_layers(siamese_model: Model, number_of_layers_to_unfroze: int, model_optimizer):
    model_layer = siamese_model.layers[2]  # layers 0 and 1 are input, model is the third
    unfroze_limit = number_of_layers_to_unfroze + number_of_layers_to_unfroze // 4 + 1
    for layer in model_layer.layers[:-unfroze_limit]:
        layer.trainable = False
    for layer in model_layer.layers[-unfroze_limit:]:
        layer.trainable = True


def data_triple_generator(datagen: ImageDataGenerator, x_im1: np.ndarray, x_im2: np.ndarray, y: np.ndarray, batch_size: int):
    for i, ((im1, label), (im2, _)) in enumerate(zip(datagen.flow(x_im1, y, batch_size=batch_size),
                                                     datagen.flow(x_im2, y, batch_size=batch_size))):
        if random.random() <= 0.5:
            yield [im1, im2], label
        else:
            yield [im2, im1], label


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
    parser.add_argument('-vrb', '--vgg-nb-block-to-remove',
                        default=0,
                        type=int,
                        dest="vgg_nb_block_to_remove")
    parser.add_argument('-o', '--out-dir',
                        default=None,
                        type=str,
                        dest="output_dir")
    parser.add_argument('-e', '--epochs-per-step',
                        default=1,
                        type=int,
                        dest="number_of_epoch")
    parser.add_argument('-d', '--dropout',
                        default=None,
                        type=float,
                        dest="dropout")
    parser.add_argument('-lr', '--learning-rate',
                        default=0.001,
                        type=float,
                        dest="learning_rate")
    parser.add_argument('-lrd', '--learning-rate-decay',
                        default=0.0,
                        type=float,
                        dest="learning_rate_decay")
    parser.add_argument('-op', '--optimizer',
                        default='adam',
                        type=str,
                        dest="optimizer")
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
                                  merge_type=args.merge_layer,
                                  layer_block_to_remove=args.vgg_nb_block_to_remove,
                                  dropout=args.dropout)
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=args.optimizer,
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

    # img_paths = glob.glob(os.path.join("data2/test/", '*/*/*.png'))
    # fit_images = np.array(map(cv2.imread, img_paths))
    datagen.fit(np.array(list(map(cv2.imread, glob.glob(os.path.join("data2/test/", '*/*/*.png'))))))
    datagen_test.fit(np.array(list(map(cv2.imread, glob.glob(os.path.join("data2/test/", '*/*/*.png'))))))

    print("fit done")

    # triple_generator = data_triple_generator(datagen, x_1_train, x_2_train, y_train, batch_size)
    triple_generator = data_triple_generator_from_dir(datagen, "data2/train", args.batch_size)
    triple_generator_test = data_triple_generator_from_dir(datagen_test, "data2/test", args.batch_size, shuffle=False)

    # train the top layer of the classifier
    history = model.fit_generator(generator=triple_generator,
                                  steps_per_epoch=TRAIN_DATASET_SIZE // args.batch_size + 1,
                                  epochs=args.number_of_epoch,
                                  verbose=1,
                                  validation_data=triple_generator_test,
                                  validation_steps=TEST_DATASET_SIZE // args.batch_size + 1,
                                  initial_epoch=0
                                  )
    # save the result for analysis
    epoch = history.epoch
    h_values = history.history.values()
    values = np.array([epoch, ] + list(h_values) + [[0] * len(epoch)])
    df_history = pd.DataFrame(data=values.T,
                              columns=["epoch", ] + list(history.history.keys()) + ['fine_tuning']
                              )

    if args.optimizer == 'adam':
        opt = Adam(lr=args.learning_rate,
                   decay=args.learning_rate_decay)
    elif args.optimizer == 'rmsprop':
        opt = RMSprop(lr=args.learning_rate,
                      decay=args.learning_rate_decay)
    else:
        raise ValueError("Optimizer argument must be one of 'adam' or 'rmsprop', not " + str(args.optmizer))

    # now do some fine tuning if asked from command line
    for i in range(1, args.fine_tuning_iteration + 1):
        unfroze_core_model_layers(model, i, opt)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['categorical_accuracy'])

        history = model.fit_generator(generator=triple_generator,
                                      steps_per_epoch=TRAIN_DATASET_SIZE // args.batch_size + 1,
                                      epochs=args.number_of_epoch,
                                      verbose=1,
                                      validation_data=triple_generator_test,
                                      validation_steps=TEST_DATASET_SIZE // args.batch_size + 1,
                                      initial_epoch=0
                                      )
        # save the result for analysis
        epoch = history.epoch
        h_values = history.history.values()
        values = np.array([epoch, ] + list(h_values) + [[i] * len(epoch)])
        df_history = df_history.append(pd.DataFrame(data=values.T,
                                                    columns=["epoch", ] + list(history.history.keys()) + ['fine_tuning']
                                                    )
                                       )

    print(args.__dict__)
    for k, v in args.__dict__.items():
        print(k, v, df_history.shape[0])
        kwargs = {k: [v] * df_history.shape[0]}
        df_history = df_history.assign(**kwargs)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        # write history to csv for late160r use

        df_history.to_csv(os.path.join(args.output_dir, 'history.csv'),
                          sep=',')
        # do some fancy plots
        # Accuracy
        plt.figure()
        plt.plot(df_history.get('epoch'), df_history.get('val_categorical_accuracy'), label='val_categorical_accuracy')
        plt.plot(df_history.get('epoch'), df_history.get('categorical_accuracy'), label='categorical_accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, 'accuracy.png'))
        # Loss
        plt.figure()
        plt.plot(df_history.get('epoch'), df_history.get('val_loss'), label='val_loss')
        plt.plot(df_history.get('epoch'), df_history.get('loss'), label='loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, 'loss.png'))

        with open(os.path.join(args.output_dir, 'args.txt'), mode='w') as f:
            f.write(' '.join(str(sys.argv)))


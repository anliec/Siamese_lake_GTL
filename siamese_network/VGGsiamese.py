import matplotlib
# solve plotting issues with matplotlib when no X connection is available
matplotlib.use('Agg')

from keras.applications import VGG16
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import numpy as np
import glob
import os
import cv2
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from siamese_network.siamese import get_siamese_layers, data_triple_generator_from_dir
from evaluator.evaluate import DatasetTester


K.set_image_dim_ordering('tf')


def get_siamese_vgg_model(image_shape=(224, 224, 3), weights='imagenet', train_from_layers: int=19,
                          merge_type='concatenate', add_batch_norm=False, layer_block_to_remove=0,
                          dropout=None):
    """
    create a VGG based, siamese network
    :param image_shape: shape of input ((224, 224, 3) if the original weight are used)
    :param weights: source of the weight to use on VGG, can by 'imagenet' or 'None' (or a path to weights)
    :param train_from_layers: number of layers to froze starting from the first layer (default 19: all)
    :param merge_type: how to merge the output of the siamese network, one of 'dot', 'multiply',
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

    top = Dense(512, activation="relu")(siamese_vgg)
    if dropout is not None:
        top = Dropout(dropout)(top)
    top = Dense(512, activation="relu")(top)
    if dropout is not None:
        top = Dropout(dropout)(top)
    top = Dense(2, activation="softmax")(top)

    return Model(inputs=[input_a, input_b], outputs=top)


def unfroze_core_model_layers(siamese_model: Model, number_of_layers_to_unfroze: int):
    """
    Get into the given siamese model to unfroze the last layers of the convectional part
    :param siamese_model: The siamese model which will be modified
    :param number_of_layers_to_unfroze: Number of layers where weight will be trainable after this function,
    starting from last layers.
    :return: None
    """
    model_layer = siamese_model.layers[2]  # layers 0 and 1 are input, model is the third
    unfroze_limit = number_of_layers_to_unfroze + number_of_layers_to_unfroze // 4 + 1
    for layer in model_layer.layers[:-unfroze_limit]:
        layer.trainable = False
    for layer in model_layer.layers[-unfroze_limit:]:
        layer.trainable = True


def save_results(path: str, history: pd.DataFrame):
    """
    Save the given history to the given path with some plots of the results
    :param path: path were all the files will be created
    :param history: a panda dataframe containing the information about the learning process, mostly the information
    from model.fit.
    :return: None
    """
    if path is not None:
        os.makedirs(path, exist_ok=True)
        # write history to csv for late160r use

        history.to_csv(os.path.join(path, 'history.csv'), sep=',')
        # do some fancy plots
        # Accuracy
        plt.figure()
        plt.plot(history.get('epoch'), history.get('val_categorical_accuracy'), label='val_categorical_accuracy')
        plt.plot(history.get('epoch'), history.get('categorical_accuracy'), label='categorical_accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(os.path.join(path, 'accuracy.png'))
        # Loss
        plt.figure()
        plt.plot(history.get('epoch'), history.get('val_loss'), label='val_loss')
        plt.plot(history.get('epoch'), history.get('loss'), label='loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(os.path.join(path, 'loss.png'))
        # Evaluate model and save results


def evaluate_model(model, tester: DatasetTester, currant_epoch: int, save_path: str, batch_size: int):
    if save_path is not None:
        # evaluate model
        result_list = tester.evaluate(model,
                                      mode="both",
                                      batch_size=batch_size,
                                      add_coordinate=True)
        file_name = "evaluation_epoch_" + str(currant_epoch) + ".pickle"
        with open(os.path.join(save_path, file_name), 'wb') as handle:
            pickle.dump(result_list, handle)


def main():
    """
    Build and train a VGG Siamese network using the provided command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size',
                        default=40,
                        type=int,
                        dest="batch_size")
    parser.add_argument('-bn', '--batch-norm',
                        default=True,
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
                        default=0.0,
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
    parser.add_argument('-dp', '--dataset-path',
                        default="./data",
                        type=str,
                        dest="data_set_path")
    args = parser.parse_args()
    if args.output_dir is not None:
        model_save_path = args.output_dir + "_models"
        os.makedirs(model_save_path, exist_ok=True)
    else:
        model_save_path = None

    # check dataset
    if not os.path.isdir(args.data_set_path):
        raise ValueError("the specified dataset directory ('{}') is not a directory".format(args.data_set_path))
    # check if left and right have the same number of images
    for data_class in ['1', '-1']:
        for dataset in ['train', 'test']:
            left_images_paths = glob.glob(os.path.join(args.data_set_path, dataset, 'left', data_class, '/*'))
            right_images_paths = glob.glob(os.path.join(args.data_set_path, dataset, 'right', data_class, '/*'))
            assert len(left_images_paths) == len(right_images_paths)

    model = get_siamese_vgg_model(add_batch_norm=args.use_batch_norm,
                                  train_from_layers=args.vgg_frozen_layer,
                                  merge_type=args.merge_layer,
                                  layer_block_to_remove=args.vgg_nb_block_to_remove,
                                  dropout=args.dropout)
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  # categorical_crossentropy with 2 labels is the same than binary_crossentropy
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

    train_images_paths = glob.glob(os.path.join(args.data_set_path, 'train/*/*/*'))
    test_images_paths = glob.glob(os.path.join(args.data_set_path, 'test/*/*/*'))
    datagen.fit(np.array(list(map(cv2.imread, train_images_paths[:200]))))
    datagen_test.fit(np.array(list(map(cv2.imread, test_images_paths[:200]))))

    # create dataset tester
    tester = DatasetTester(dataset_path=args.data_set_path,
                           datagen_test=datagen_test)

    if len(test_images_paths) % 2 == 1 or len(train_images_paths) % 2 == 1:
        raise ValueError("The dataset is probably incorrect as it contain an even number of images")
    number_of_test_pair = len(test_images_paths) // 2
    number_of_train_pair = len(train_images_paths) // 2

    print("fit done")

    # init triple generator (image 1, image 2, label)
    triple_generator = data_triple_generator_from_dir(datagen,
                                                      os.path.join(args.data_set_path, 'train'),
                                                      args.batch_size)
    triple_generator_test = data_triple_generator_from_dir(datagen_test,
                                                           os.path.join(args.data_set_path, 'test'),
                                                           args.batch_size,
                                                           shuffle=False)

    # train the top layer of the classifier
    history = model.fit_generator(generator=triple_generator,
                                  steps_per_epoch=number_of_train_pair // args.batch_size + 1,
                                  epochs=args.number_of_epoch,
                                  verbose=1,
                                  validation_data=triple_generator_test,
                                  validation_steps=number_of_test_pair // args.batch_size + 1,
                                  initial_epoch=0
                                  )
    # save the result for analysis
    epoch = history.epoch
    h_values = history.history.values()
    values = np.array([epoch, ] + list(h_values) + [[0] * len(epoch)])
    df_history = pd.DataFrame(data=values.T,
                              columns=["epoch", ] + list(history.history.keys()) + ['fine_tuning']
                              )
    # save current model to disk if a path was specified
    if model_save_path is not None:
        model.save(os.path.join(model_save_path, "model0.h5"), overwrite=True)
    evaluate_model(model, tester, args.number_of_epoch, model_save_path, args.batch_size)

    # update optimizer for fine tuning
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
        # unfroze the model a bit more
        unfroze_core_model_layers(model, i)

        # compile the model to apply the modifications
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['categorical_accuracy'])

        # fit the model with this new freedom
        history = model.fit_generator(generator=triple_generator,
                                      steps_per_epoch=number_of_train_pair // args.batch_size + 1,
                                      epochs=(i + 1) * args.number_of_epoch,
                                      verbose=1,
                                      validation_data=triple_generator_test,
                                      validation_steps=number_of_test_pair // args.batch_size + 1,
                                      initial_epoch=i * args.number_of_epoch
                                      )
        # save the result for analysis
        epoch = history.epoch
        h_values = history.history.values()
        values = np.array([epoch, ] + list(h_values) + [[i] * len(epoch)])
        df_history = df_history.append(pd.DataFrame(data=values.T,
                                                    columns=["epoch"] + list(history.history.keys()) + ['fine_tuning']))
        # save current model to disk if a path was specified
        if model_save_path is not None:
            model.save(os.path.join(model_save_path, "model" + str(i) + ".h5"), overwrite=True)
        evaluate_model(model, tester, args.number_of_epoch * i, model_save_path, args.batch_size)

    # add the training argument to history, to make filtering easier when all
    # the different history will be merged together.
    for k, v in args.__dict__.items():
        kwargs = {k: [v] * df_history.shape[0]}
        df_history = df_history.assign(**kwargs)

    save_results(args.output_dir, df_history)


if __name__ == '__main__':
    main()


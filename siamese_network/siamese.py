from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Concatenate, BatchNormalization, Subtract,\
    Multiply
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import random
import numpy as np


def get_siamese_model(src_model: Model, input_shape: tuple, add_batch_norm: bool=False, merge_type: str='concatenate'):
    """
    Create a siamese model from the given parameters
    :param src_model: model used for the siamese part, for example if vgg is provided here this will build a siamese
        network where using vgg to transform the images
    :param input_shape: shape of the input of the given model
    :param add_batch_norm: if batch normalisation must be added around the merge layers
    :param merge_type: how to merge the output of the siamese network, one of 'dot', 'multiply',
        'subtract', 'l1', 'l2' or 'concatenate'.
    :return: the siamese model
    """
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    siamese = get_siamese_layers(src_model, input_a, input_b, add_batch_norm, merge_type)

    model = Model([input_a, input_b], siamese)
    return model


def get_siamese_layers(src_model: Model, input_a, input_b, add_batch_norm=False, merge_type='concatenate'):
    """
    Create a set of layers needed to construct a siamese model (siamese structure plus merging method)
    :param src_model:  model used for the siamese part, for example if vgg is provided here this will build a siamese
        network where using vgg to transform the images
    :param input_a: input layer for one side of the model
    :param input_b: input layer for the other side of the model
    :param add_batch_norm: if batch normalisation must be added around the merge layers
    :param merge_type: how to merge the output of the siamese network, one of 'dot', 'multiply',
        'subtract', 'l1', 'l2' or 'concatenate'.
    :return: output layer of siamese part
    """
    processed_a = src_model(input_a)
    processed_b = src_model(input_b)
    if add_batch_norm:
        processed_a = BatchNormalization(name='processed_a_normalization')(processed_a)
        processed_b = BatchNormalization(name='processed_b_normalization')(processed_b)

    if merge_type == 'concatenate':
        siamese = Concatenate(name='concatenate_merge')([processed_a, processed_b])
        siamese = Flatten(name='concatenate_merge_flatten')(siamese)
    elif merge_type == 'dot':
        siamese = Multiply(name='dot_merge_multiply')([processed_a, processed_b])
        siamese = Lambda(lambda x: K.sum(x, axis=(1, 2)), name='dot_merge_sum')(siamese)
    elif merge_type == 'subtract':
        siamese = Subtract(name='subtract_merge')([processed_a, processed_b])
        siamese = Flatten(name='subtract_merge_flatten')(siamese)
    elif merge_type == 'l1':
        siamese = Subtract(name='l1_merge_subtract')([processed_a, processed_b])
        siamese = Lambda(lambda x: K.abs(x), name='l1_merge_abs')(siamese)
        siamese = Flatten(name='l1_merge_flatten')(siamese)
    elif merge_type == 'l2':
        siamese = Subtract(name='l2_merge_subtract')([processed_a, processed_b])
        siamese = Lambda(lambda x: K.pow(x, 2), name='l2_merge_square')(siamese)
        siamese = Flatten(name='l2_merge_flatten')(siamese)
    elif merge_type == 'multiply':
        siamese = Multiply(name='multiply_merge')([processed_a, processed_b])
        siamese = Flatten(name='multiply_merge_flatten')(siamese)
    else:
        raise ValueError("merge_type value incorrect, was " + str(merge_type) + " and not one of 'concatenate', 'dot', "
                         "'subtract', 'l1', 'l2' or 'multiply'")

    if add_batch_norm:
        siamese = BatchNormalization(name='merge_normalisation')(siamese)

    return siamese


def data_triple_generator_from_dir(datagen: ImageDataGenerator, dataset_dir, batch_size: int, seed=6,
                                   save_to_dir: str=None, shuffle: bool=True, include_label: bool=True):
    """
    Iterator using keras ImageDataGenerator to iterate the dataset (without loading it to memory) and provide paires
    of images and their corresponding label if needed
    :param datagen: Data generator which used for both image of the pair
    :param dataset_dir: Path to the directory containing the dataset (including the test / train part)
    :param batch_size: Size of the generated batch
    :param seed: Random seed of the generator
    :param save_to_dir: Path where the augmented images will be saved, None to prevent saving
    :param shuffle: True to give the paires in a random order, False to provide them the same way the os give them
    :param include_label: set to True to add the label to image pair, False to only provide the image pair (useful for
        prediction)
    """
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
        if random.random() <= 0.5:
            if include_label:
                yield [im1, im2], label1
            else:
                yield [im1, im2]
        else:
            if include_label:
                yield [im2, im1], label1
            else:
                yield [im2, im1]



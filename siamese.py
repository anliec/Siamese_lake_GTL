from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Concatenate, BatchNormalization, Subtract,\
    Multiply
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import random
import numpy as np


def get_siamese_model(src_model: Model, input_shape: tuple, add_batch_norm=False, merge_type='concatenate'):
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    siamese = get_siamese_layers(src_model, input_a, input_b, add_batch_norm, merge_type)

    model = Model([input_a, input_b], siamese)
    return model


def get_siamese_layers(src_model: Model, input_a, input_b, add_batch_norm=False, merge_type='concatenate'):
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


def data_triple_generator(datagen: ImageDataGenerator, x_im1: np.ndarray, x_im2: np.ndarray, y: np.ndarray, batch_size: int):
    for i, ((im1, label), (im2, _)) in enumerate(zip(datagen.flow(x_im1, y, batch_size=batch_size),
                                                     datagen.flow(x_im2, y, batch_size=batch_size))):
        if random.random() <= 0.5:
            yield [im1, im2], label
        else:
            yield [im2, im1], label


def data_triple_generator_from_dir(datagen: ImageDataGenerator, dataset_dir, batch_size: int, seed=6,
                                   save_to_dir: str=None, shuffle=True, include_label=True):
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



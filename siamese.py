"""Trains a Siamese MLP on pairs of digits from the MNIST dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).
# References
- Dimensionality Reduction by Learning an Invariant Mapping
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
Gets to 97.2% test accuracy after 20 epochs.
2 seconds per epoch on a Titan X Maxwell GPU

src: https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Concatenate, BatchNormalization, Subtract, Dot,\
    Multiply
from keras.optimizers import RMSprop
from keras import backend as K


def get_siamese_model(src_model: Model, input_shape: tuple, add_batch_norm=False, merge_type='concatenate'):
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = src_model(input_a)
    processed_b = src_model(input_b)
    if add_batch_norm:
        processed_a = BatchNormalization()(processed_a)
        processed_b = BatchNormalization()(processed_b)

    if merge_type == 'concatenate':
        siamese = Concatenate()([processed_a, processed_b])
    elif merge_type == 'dot':
        siamese = Dot(axes=-1,
                      normalize=True
                      )([processed_a, processed_b])
    elif merge_type == 'subtract':
        siamese = Subtract()([processed_a, processed_b])
    elif merge_type == 'multiply':
        siamese = Multiply()([processed_a, processed_b])
    else:
        raise ValueError("merge_type value incorrect, was " + str(merge_type) + " and not one of 'concatenate', 'dot', "
                         "'subtract' or 'multiply'")
    siamese = Flatten()(siamese)

    model = Model([input_a, input_b], siamese)

    return model

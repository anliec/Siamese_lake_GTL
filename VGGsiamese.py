from keras.applications import VGG16
from keras.layers import Dense, Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from dataset_loader import load_data_set
from siamese import get_siamese_model


def get_siamese_vgg_model(image_shape=(224, 224, 3)):
    input_a = Input(image_shape)
    input_b = Input(image_shape)

    vgg_base = VGG16(include_top=False, input_shape=image_shape, weights='imagenet')

    siamese_vgg_model = get_siamese_model(vgg_base, image_shape)
    siamese_vgg_model.trainable = False

    top = Dense(512, activation="relu")(siamese_vgg_model([input_a, input_b]))
    top = Dense(128, activation="relu")(top)
    top = Dense(2, activation="softmax")(top)

    return Model(inputs=[input_a, input_b], outputs=top)


def data_triple_generator(datagen: ImageDataGenerator, x_im1: np.ndarray, x_im2: np.ndarray, y: np.ndarray, batch_size: int):
    for i, ((im1, label), (im2, _)) in enumerate(zip(datagen.flow(x_im1, y, batch_size=batch_size),
                                                     datagen.flow(x_im2, y, batch_size=batch_size))):
        yield [im1, im2], label
        # add vertical flip on both images ? randomly switch images ?


if __name__ == '__main__':
    batch_size = 32
    train_ratio = 0.8

    x_1, x_2, y = load_data_set()
    train_size = int(train_ratio * len(y))
    x_1_train, x_2_train, y_train = x_1[:train_size], x_2[:train_size], y[:train_size]
    x_1_test, x_2_test, y_test = x_1[train_size:], x_2[train_size:], y[train_size:]
    
    model = get_siamese_vgg_model()
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    datagen = ImageDataGenerator(  # add shear_range ?
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=[0.6, 1.1],
        fill_mode='reflect',
        horizontal_flip=False,
        vertical_flip=False)

    datagen.fit(np.append(x_1_train, x_2_train, axis=0))

    triple_generator = data_triple_generator(datagen, x_1_train, x_2_train, y_train, batch_size)
    model.fit_generator(generator=triple_generator,
                        steps_per_epoch=len(y) // batch_size,
                        epochs=1,
                        verbose=1,
                        validation_data=([x_1_test, x_2_test], y_test)
                        )


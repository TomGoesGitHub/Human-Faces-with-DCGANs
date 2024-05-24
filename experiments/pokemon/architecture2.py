from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Reshape
from tensorflow.keras.layers import Flatten, BatchNormalization, Dense, Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SHAPE = [64,64,3]

def get_generator():
    """
    This method returns a generator model that generates noise that will fool the discriminator.
    :return:
    """
    generator = Sequential()
    generator.add(Dense(units=4 * 4 * 512,
                        kernel_initializer='glorot_uniform',
                        input_shape=(1, 1, 100)))
    generator.add(Reshape(target_shape=(4, 4, 512)))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=256, kernel_size=(5, 5),
                                    strides=(2, 2), padding='same',
                                    data_format='channels_last',
                                    kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=128, kernel_size=(5, 5),
                                    strides=(2, 2), padding='same',
                                    data_format='channels_last',
                                    kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=64, kernel_size=(5, 5),
                                    strides=(2, 2), padding='same',
                                    data_format='channels_last',
                                    kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=3, kernel_size=(5, 5),
                                    strides=(2, 2), padding='same',
                                    data_format='channels_last',
                                    kernel_initializer='glorot_uniform'))
    generator.add(Activation('tanh'))

    # optimizer = Adam(lr=0.00015, beta_1=0.5)
    # generator.compile(loss='binary_crossentropy',
    #                     optimizer=optimizer,
    #                     metrics=None)

    return generator

def get_discriminator():
    """
    Learns to identify if a given image is similar to that of the training data or not
    :return:
    """
    discriminator = Sequential()
    discriminator.add(Conv2D(filters=64, kernel_size=(5, 5),
                                strides=(2, 2), padding='same',
                                data_format='channels_last',
                                kernel_initializer='glorot_uniform',
                                input_shape=IMG_SHAPE))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(filters=128, kernel_size=(5, 5),
                                strides=(2, 2), padding='same',
                                data_format='channels_last',
                                kernel_initializer='glorot_uniform'))
    discriminator.add(BatchNormalization(momentum=0.5))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(filters=256, kernel_size=(5, 5),
                                strides=(2, 2), padding='same',
                                data_format='channels_last',
                                kernel_initializer='glorot_uniform'))
    discriminator.add(BatchNormalization(momentum=0.5))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(filters=512, kernel_size=(5, 5),
                                strides=(2, 2), padding='same',
                                data_format='channels_last',
                                kernel_initializer='glorot_uniform'))
    discriminator.add(BatchNormalization(momentum=0.5))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Flatten())
    discriminator.add(Dense(1))
    discriminator.add(Activation('sigmoid'))

    # optimizer = Adam(lr=0.0002, beta_1=0.5)
    # discriminator.compile(loss='binary_crossentropy',
    #                         optimizer=optimizer,
    #                         metrics=None)

    return discriminator


discriminator = get_discriminator()
generator = get_generator()
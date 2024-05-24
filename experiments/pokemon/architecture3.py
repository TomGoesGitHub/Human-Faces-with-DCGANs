'''
Architecture from the paper Redford 2016 - Unsupervised Represenation Learning With 
Deep Convolutional Generative Adversarial Networks.
https://arxiv.org/pdf/1511.06434.pdf
'''

import tensorflow as tf

IMG_SHAPE = [64,64,3]
LATENT_DIM = 100


def _build_upsampling_cnn_block(filters, name):
    return tf.keras.Sequential(
                layers=[
                    tf.keras.layers.Conv2DTranspose(filters, kernel_size=5, strides=2, padding='same',
                                                    activation=None, use_bias=False),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                ],
                name=name 
           )

def _build_downsampling_cnn_block(filters, name):
    return tf.keras.Sequential(
                layers=[
                    tf.keras.layers.Conv2D(filters, kernel_size=5, strides=2, padding='same',
                                           activation=None, use_bias=False),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                ],
                name=name
           )


generator = tf.keras.Sequential(
    layers = [
        # input
        tf.keras.layers.Input(shape=LATENT_DIM),
        tf.keras.layers.Reshape(target_shape=[1,1, LATENT_DIM]),
        tf.keras.layers.Conv2DTranspose(filters=1024, kernel_size=4, padding='valid',
                                        strides=1, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        # cnn
        *[_build_upsampling_cnn_block(filters=f, name=f'cnn_block_{i}')
          for i, f in enumerate([512, 256, 128], start=1)],
        # output (note: no Batch-Normalization in the generator output layer)
        tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=5, padding='same',
                                        strides=2, use_bias=False),
        tf.keras.layers.Activation('tanh'),

    ],
    name = 'generator'
)


discriminator = tf.keras.Sequential(
    layers = [
        # input (note: no Batch-Normalization in the discriminator input layer)
        tf.keras.layers.Input(shape=IMG_SHAPE),
        # cnn
        *[_build_downsampling_cnn_block(filters=f, name=f'cnn_block_{i}')
          for i, f in enumerate([128, 256, 512, 1024], start=1)], 
        # output
        tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding='valid', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Activation('sigmoid')
    ],
    name = 'discriminator'
)


if __name__ == '__main__':
    generator.summary()
    discriminator.summary()
    print(generator.input_shape)
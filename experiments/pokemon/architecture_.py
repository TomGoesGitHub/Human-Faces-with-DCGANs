import math
import tensorflow as tf

IMG_SHAPE = [64,64,3]

def _build_upsampling_cnn_block(filters, name):
    return tf.keras.Sequential(
                layers=[
                    tf.keras.layers.Conv2D(filters, kernel_size=5, padding='same', activation=None),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.UpSampling2D()
                ],
                name=name 
           )

def _build_downsampling_cnn_block(filters, name):
    return tf.keras.Sequential(
                layers=[
                    tf.keras.layers.Conv2D(filters, kernel_size=5, padding='same', activation=None),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.AveragePooling2D()
                ],
                name=name 
           )


generator = tf.keras.Sequential(
    layers=[
        # embedding
        tf.keras.layers.Input(shape=math.prod([4,4,512]), dtype=tf.float32),
        tf.keras.layers.Reshape(target_shape=[4,4,512]),
        # CNN
        *[_build_upsampling_cnn_block(f, name=f'cnn_block_{i}')
          for i,f in enumerate([256, 128, 64, 32], start=1)],
        # final image
        tf.keras.layers.Conv2D(filters=IMG_SHAPE[-1], kernel_size=3, activation='tanh', padding='same'),
        #tf.keras.layers.Resizing(height=IMG_SHAPE[0], width=IMG_SHAPE[1])
    ],
    name='Generator'
)


discriminator = tf.keras.Sequential(
    layers=[
        tf.keras.layers.Input(shape=IMG_SHAPE, dtype=tf.float32),
        # CNN
        *[_build_downsampling_cnn_block(f, name=f'cnn_block_{i}')
          for i,f in enumerate([32, 64, 128, 256, 512])],
        # Fully Conectected
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ],
    name='Discriminator'
)
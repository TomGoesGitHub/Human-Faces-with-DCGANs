import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class PokemonDataLoader(tf.keras.utils.Sequence):
    def __init__(self, files, batch_size, augmentation=False):
        self.files = files
        self.batch_size = batch_size
        self.augmentation = augmentation
    
    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.files))
        files = self.files[low:high]
        imgs = [plt.imread(f) for f in files]
        imgs = tf.convert_to_tensor(imgs, dtype=tf.float32)
        if self.augmentation:
            imgs = self.apply_augmentation(imgs)
        imgs = (imgs - (255/2)) / (255/2) # rescale to range [0,1]
        return [imgs, tf.constant([])] # x,y
    
    def apply_augmentation(self, imgs):
        # augment orientation
        imgs = tf.keras.layers.RandomFlip(mode='horizontal')(imgs, training=True)
        imgs = tf.keras.layers.RandomRotation(factor=0.1, fill_mode='constant', fill_value=255.)(imgs, training=True)
        
        # augment colors
        full_white = 255*tf.ones_like(imgs)
        is_white = tf.stack(3*[tf.reduce_all((full_white-imgs<=20), axis=-1)], axis=-1)
        color_shift = tf.stack([tf.ones(imgs.shape[1:]) * tf.random.uniform(shape=[3], minval=0, maxval=255)
                                for _ in range(len(imgs))])
        imgs = (imgs + color_shift) % 255 # bring back into valid range
        imgs = tf.clip_by_value(imgs, 50, 200) # ensure 'natural' colors
        imgs = tf.where(is_white, full_white, imgs) # make white pixels white again
        return imgs


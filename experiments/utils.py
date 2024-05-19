import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class VisualizeGeneratedFakesCallback(tf.keras.callbacks.Callback):#
    def __init__(self, z_distribution, freq=1, pixelvalue_range=(-1,1), save_dir=None):
        self._z = z_distribution.sample(sample_shape=35)
        self.pixelvalue_range = pixelvalue_range
        self.freq = freq
        self.save_dir = save_dir

        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.freq == 0:
            fake_imgs = self.model(self._z)
            pvr = self.pixelvalue_range # alias for readability 
            fake_imgs = (fake_imgs - pvr[0]) / (pvr[1] - pvr[0]) # rescale into interval [0, 1]
            fig, axes = plt.subplots(ncols=7, nrows=5)
            for ax, img in zip(np.ravel(axes), fake_imgs):
                ax.imshow(img)
                ax.axis('off')
            plt.tight_layout()
            plt.show()
            if self.save_dir:
                file_name = os.path.join(self.save_dir, f'{epoch=}.png')
                plt.savefig(file_name)
            plt.close()

        

    
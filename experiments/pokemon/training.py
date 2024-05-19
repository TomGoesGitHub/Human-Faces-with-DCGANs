import os
import sys

sys.path.append(os.path.join(os.pardir, os.pardir))
from gan import GenerativeAdversarialNetwork
from experiments.pokemon.architecture import generator, discriminator
from experiments.utils import VisualizeGeneratedFakesCallback
from experiments.pokemon.utils import PokemonDataLoader

def run_training(gan, data_dir, result_dir):
    filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    dataloader = PokemonDataLoader(filepaths, batch_size=64, augmentation=False)

    callbacks=[VisualizeGeneratedFakesCallback(z_distribution=gan.z_distribution, save_dir=result_dir)]
    gan.fit(x=dataloader, callbacks=callbacks, epochs=100)

if __name__ =='__main__':
    DATA_DIR = 'D:\DATASETS\Pokemon\pokemon_jpg_64x64'
    RESULT_DIR = 'results'
    # todo tmp
    import tensorflow as tf
    import matplotlib.pyplot as plt
    filepaths = [os.path.join(DATA_DIR, fn) for fn in os.listdir(DATA_DIR)]
    some_batch, _ = PokemonDataLoader(filepaths, batch_size=64, augmentation=False)[0]
    # todo tmp end
    gan = GenerativeAdversarialNetwork(generator, discriminator,
                                z_shape=generator.input_shape[1:],
                                x_shape=discriminator.input_shape[1:],
                                historical_averaging=True,
                                feature_matching_idx=2,
                                virtual_normalization_batch=some_batch)
    gan.summary()
    gan.compile()
    run_training(gan, DATA_DIR, RESULT_DIR)

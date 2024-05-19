import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from gan_utils import HistoricalAveraging, FeatureMatching, VirtualBatchNorm

@tf.keras.utils.register_keras_serializable()
class GenerativeAdversarialNetwork(tf.keras.Model):
    def __init__(# Vanilla GAN
                 self,
                 generator, discriminator,
                 z_shape, x_shape,
                 k=1,
                 # improved techniques for GAN-Training
                 label_smooting=None,
                 historical_averaging=False,
                 feature_matching_idx=None,
                 virtual_normalization_batch=None,
                 loss_weighting={
                     'historical_averaging': 1.,
                     'feature_matching': 1.,
                 },
                 **kwargs):
        super().__init__(**kwargs)
        # Vanilla GAN
        self.z_shape = z_shape
        self.x_shape = x_shape
        self.k = k # number of discriminator-updates per generator-update
        self.build(generator, discriminator)
        
        # improved techniques for GAN-Training
        self.label_smoothing = label_smooting if label_smooting is not None else (0,1)
        self.virtual_batch_norm = VirtualBatchNorm(virtual_normalization_batch, model=self.discriminator) \
                                  if virtual_normalization_batch is not None else None
        self.feature_matching = FeatureMatching(model=self.discriminator,
                                                layer_index=feature_matching_idx,
                                                coef=loss_weighting['feature_matching']) \
                                if feature_matching_idx else None
        self.historical_averaging_D = HistoricalAveraging(model=self.discriminator,
                                                          coef=loss_weighting['historical_averaging']) \
                                      if historical_averaging else None
        self.historical_averaging_G = HistoricalAveraging(model=self.generator,
                                                          coef=loss_weighting['historical_averaging']) \
                                      if historical_averaging else None
        self.loss_weighting = loss_weighting

    def build(self, generator, discriminator):
        base_distribution = tfd.Normal(loc=tf.zeros(shape=self.z_shape), scale=tf.ones(shape=self.z_shape))
        self.z_distribution = tfd.Independent(distribution=base_distribution, reinterpreted_batch_ndims=1)
        self.generator = generator
        self.discriminator = discriminator
        super().build(input_shape=[None, *self.z_shape])

    def summary(self):
        self.generator.summary(expand_nested=True)
        self.discriminator.summary(expand_nested=True)
        super().summary(expand_nested=False)

    def generate(self, batch_size):
        z = self.z_distribution.sample(sample_shape=[batch_size, *self.z_shape])
        return self.generator(z)
    
    def call(self, inputs, training=False):
        z = inputs
        return self.generator(z, training)

    def compile(self,
                optimizer_discriminator=tf.keras.optimizers.Adam(learning_rate=1e-4),
                optimizer_generator=tf.keras.optimizers.Adam(learning_rate=1e-4)):
        super().compile()
        self.optimizer_discriminator = optimizer_discriminator
        self.optimizer_generator = optimizer_generator
        self.train_step_index = tf.Variable(0, trainable=False, dtype=tf.int32)
    
    tf.config.run_functions_eagerly(True) # Note: uncomment for debugging
    #@tf.function
    def train_step(self, data):
        x, _ = data
        self.train_step_index.assign_add(1)

        result_dict = self.update_discriminator(x)
        if self.train_step_index % self.k == 0:
            self.update_generator(x)
        return result_dict

    def update_discriminator(self, x):        
        batch_size = tf.shape(x)[0]
        z = self.z_distribution.sample(sample_shape=[batch_size])
        x_fake = self.generator(z, training=True)
        cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        with tf.GradientTape() as tape:
            # vanilla loss
            logits_real = self.discriminator(x, training=True)
            logits_fake = self.discriminator(x_fake, training=True)
            targets_real = tf.ones_like(logits_real) * self.label_smoothing[1]
            targets_fake = tf.ones_like(logits_real) * self.label_smoothing[0]
            ce_real = cross_entropy_loss(targets_real, logits_real)
            ce_fake = cross_entropy_loss(targets_fake, logits_fake)
            vanilla_loss = 0.5 * (ce_real + ce_fake)
            
            # additional penalty terms
            penalty_historical_averaging = self.historical_averaging_D() if self.historical_averaging_D else 0
            
            loss = vanilla_loss + penalty_historical_averaging

        gradients = tape.gradient(target=loss, sources=self.discriminator.trainable_variables)
        self.optimizer_discriminator.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
        
        return {'Cross-Entropy': vanilla_loss,
                'E[p(y=real|x=real)]': tf.reduce_mean(logits_real),
                'E[p(y=fake|x=fake)]': 1-tf.reduce_mean(logits_fake),
                'Historical-Averaging-Penalty': penalty_historical_averaging,}


    def update_generator(self, x):
        batch_size = tf.shape(x)[0]
        z = self.z_distribution.sample(sample_shape=[batch_size])
        cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
        if self.virtual_batch_norm:
            self.virtual_batch_norm()
            # note: Following the sugestion in Salimans-Paper, VBN is placed here: "VBN is
            # computationally expensive because it requires running forward propagation on two minibatches of
            # data, so we use it only in the generator network."
        
        with tf.GradientTape() as tape:
            # vanilla loss
            x_fake = self.generator(z, training=True)
            logits_fake = self.discriminator(x_fake, training=False)
            ce_fake = cross_entropy_loss(tf.ones_like(logits_fake), logits_fake)
            vanilla_loss = ce_fake
            # note: Swapped target-labels in order to maximize log(D(G(z))
            # From Goodfellow-Paper: "Early in learning, when G is poor, D can reject samples with high
            # confidence because they are clearly different from the training data. In this case, log(1 − D(G(z)))
            # saturates. Rather than training G to minimize log(1 − D(G(z))) we can train G to maximize log D(G(z))."
        
            # additional penalty terms
            penalty_historical_averaging = self.historical_averaging_G() if self.historical_averaging_G else 0
            penatly_feature_matching = self.feature_matching(x, x_fake) if self.feature_matching else 0

            loss = vanilla_loss + penalty_historical_averaging + penatly_feature_matching

        gradients = tape.gradient(target=loss, sources=self.generator.trainable_variables)
        self.optimizer_generator.apply_gradients(zip(gradients, self.generator.trainable_variables))
        # todo: I do not know how result stats are calculated during one epoch of training if G is only updated every k steps

                




        


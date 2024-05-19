import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

#@tf.keras.utils.register_keras_serializable()
class GenerativeAdversarialNetwork(tf.keras.Model):
    def __init__(self, generator, discriminator, z_shape, x_shape,
                k=5, **kwargs):
        super().__init__(**kwargs)
        self.z_shape = z_shape
        self.x_shape = x_shape
        self.k = k # number of discriminator-updates per generator-update
        self.build(generator, discriminator)

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
        self.train_step_index = tf.Variable(0, trainable=False)
    
    #tf.config.run_functions_eagerly(True)
    def train_step(self, data):
        x, _ = data
        cross_entropy = self.update_discriminator(x)
        tf.cond(self.train_step_index % self.k == 0,
                true_fn=(lambda: self.update_generator(x)),
                false_fn=(lambda: None) # do nothing
        )
        self.train_step_index.assign_add(1)
        return {'Categorical Cross-Entropy': cross_entropy}

    #@tf.function()
    def update_discriminator(self, x):
        batch_size = tf.shape(x)[0]
        z = self.z_distribution.sample(sample_shape=[batch_size])
        x_fake = self.generator(z, training=True)
        cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        with tf.GradientTape() as tape:
            logits_real = self.discriminator(x, training=True)
            logits_fake = self.discriminator(x_fake, training=True)
            ce_real = cross_entropy_loss(0.9*tf.ones_like(logits_real), logits_real) # target is 1
            ce_fake = cross_entropy_loss(tf.zeros_like(logits_fake), logits_fake) # target is 0
            loss = 0.5 * (ce_real + ce_fake)
            # log_probs_real = tf.math.log(self.discriminator(x, training=True))
            # log_probs_fake = tf.math.log(1-self.discriminator(x_fake, training=True))
            # cross_entropy = tf.reduce_mean(-log_probs_real-log_probs_fake)
            # loss = cross_entropy
        gradients = tape.gradient(target=loss, sources=self.discriminator.trainable_variables)

        self.optimizer_discriminator.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
        return loss

    #@tf.function()
    def update_generator(self, x):
        batch_size = tf.shape(x)[0]
        z = self.z_distribution.sample(sample_shape=[batch_size])
        cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        with tf.GradientTape() as tape:
            x_fake = self.generator(z, training=True)
            logits_fake = self.discriminator(x_fake, training=True)
            ce_fake = cross_entropy_loss(tf.ones_like(logits_fake), logits_fake)
            loss = ce_fake
            # loss = (-1) * tf.reduce_mean(tf.math.log(self.discriminator(x_fake, training=False)))
            # Note: From Goodfellow-Paper: Early in learning, when G is poor, D can reject samples with high
            # confidence because they are clearly different from the training data. In this case, log(1 − D(G(z)))
            # saturates. Rather than training G to minimize log(1 − D(G(z))) we can train G to maximize log D(G(z)).

        gradients = tape.gradient(target=loss, sources=self.generator.trainable_variables)
        self.optimizer_generator.apply_gradients(zip(gradients, self.generator.trainable_variables))


        


import tensorflow as tf

class HistoricalAveraging(tf.Module):
    def __init__(self, model, coef=1):
        self.coef = coef
        self.model = model
        self.moving_average = [tf.Variable(x+1e-6) for x in self.model.trainable_variables]
        # note: +1e-6 is for numerical stability since global l2-norm is not differentiable at 0
        self.n = tf.Variable(1.)
        self.decay = tf.Variable(0.99)

    def __call__(self):
        # update moving average in online fashion
        zipped = zip(self.moving_average, self.model.trainable_variables)
        for ma, v in zipped:
            #ma.assign((v+self.n*ma)/(self.n+1))
            ma.assign(self.decay * ma + (1 - self.decay) * v)
        self.n.assign_add(1)

        # calculate loss
        zipped = zip(self.moving_average, self.model.trainable_variables)
        diff = [ma-v for ma, v in zipped]
        historical_averaging_loss = self.coef * tf.linalg.global_norm(diff)**2
        return historical_averaging_loss
    
class FeatureMatching(tf.Module):
    '''https://arxiv.org/pdf/1606.03498.pdf'''
    def __init__(self, model, layer_index, coef=1):
        self.coef = coef
        self.layer_index = layer_index

        assert isinstance(model, tf.keras.Sequential)
        self.model = model
    
    def _run_model_up_to_intermediate_layer(self, input):
        x = input
        for layer in self.model.layers[:self.layer_index+1]:
            x = layer(x, training=False)
        output = x
        return output

    def __call__(self, x_real, x_fake):
        intermediate_real = self._run_model_up_to_intermediate_layer(x_real)
        intermediate_fake = self._run_model_up_to_intermediate_layer(x_fake)
        diff = tf.reduce_mean(intermediate_real, axis=0) - tf.reduce_mean(intermediate_fake, axis=0)
        feature_matching_loss = self.coef * tf.norm(diff)**2
        return feature_matching_loss

class VirtualBatchNorm(tf.Module):
    '''
    Batch Normalization causes the output of a neural network for an input example x to be highly
    dependent on several other inputs x' in the same minibatch. To avoid this problem
    we introduce virtual batch normalization (VBN), in which each example x is normalized based on
    the statistics collected on a reference batch of examples that are chosen once and fixed at the start
    of training, and on x itself. The reference batch is normalized using only its own statistics. VBN is
    computationally expensive because it requires running forward propagation on two minibatches of
    data.
    https://arxiv.org/pdf/1606.03498.pdf
    '''
    def __init__(self, virtual_batch, model):
        self.virtual_batch = tf.constant(virtual_batch)
        self.model = model
        self.freeze_or_unfreeze(is_trainable_BN=False, is_trainable_others=True) # freeze BatchNorm layers

    def _print_trainablility_recursion(self, layer):
        if isinstance(layer, tf.keras.Model):
            for layer in layer.layers:
                self._print_trainablility_recursion(layer)
        else:
            print(layer.name, layer.trainable)

    def print_trainablility(self): # for debugging
        self._print_trainablility_recursion(self.model)

    def _freeze_or_unfreeze_recursion(self, layer, is_trainable_BN, is_trainable_others):
        if isinstance(layer, tf.keras.Model):
            for layer in layer.layers:
                self._freeze_or_unfreeze_recursion(layer, is_trainable_BN, is_trainable_others)
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = is_trainable_BN
        else:
            layer.trainable = is_trainable_others

    def freeze_or_unfreeze(self, is_trainable_BN, is_trainable_others):
        self._freeze_or_unfreeze_recursion(self.model, is_trainable_BN, is_trainable_others)
    
    def __call__(self):
        # freeze non-BatchNorm-layers and unfreeze BatchNorm-layers
        self.freeze_or_unfreeze(is_trainable_BN=True, is_trainable_others=False)

        # Train Batch-Norm layers on the virtual batch only (extra forward-pass)
        _ = self.model(self.virtual_batch, training=True)

        # unfreeze non-BatchNorm-layers and freeze BatchNorm-layers
        self.freeze_or_unfreeze(is_trainable_BN=False, is_trainable_others=True)

        





        
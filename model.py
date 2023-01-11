# -*- coding: utf-8 -*-
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import numpy as np
# chyba gotowe do pierwszych testow

class BayesianRNN(tf.keras.Model):
    def __init__(self, options, place_cells, kl_weight=1.0):
        super().__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells


        self.i2h = tf.keras.layers.Dense(self.Ng, use_bias=True, activation=None)
        self.h2h = tf.keras.layers.Dense(self.Ng, use_bias=True, activation=None)
        self.p2h = tf.keras.layers.Dense(self.Ng, use_bias=True, activation=None)#, input_shape=(None,self.Np))
        self.h2o = tf.keras.layers.Dense(self.Np, use_bias=True, activation=None)

        #self.softmax = tf.keras.layers.Softmax()
        self.build_and_compile()


    def build_and_compile(self, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), run_eagerly=False,
                          input_shape=(None, 2), prev_weights=None):
        #self.build(input_shape=input_shape)
        self.compile(optimizer=optimizer, loss=None, run_eagerly=run_eagerly)
    def call(self, inputs):

        g, _ = self.g(inputs)
        return self.h2o(g)

    def g(self, inputs):
        v, p0 = inputs
        outputs = list()

        recurrent_state = self.p2h(p0)
        for t in range(self.sequence_length):
            init_state = self.h2h(recurrent_state)
            input_encoded = self.i2h(v[:,t])
            output = (input_encoded + init_state)
            output = tf.nn.relu(output)
            recurrent_state = output  # here output is of size Ng, not yet Np
            outputs.append(output)
        outputs = tf.convert_to_tensor(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
        return outputs, outputs

    def compute_loss(self, inputs, pc_outputs, pos):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        y = pc_outputs
        preds = self.call(inputs)
        yhat = tf.nn.softmax(preds)
        loss = -tf.math.reduce_sum((y * tf.math.log(yhat)), -1)
        loss = tf.math.reduce_mean(loss)

        # Weight regularization
        # Maybe in the future, but it should be handled by prior
        #loss += self.weight_decay * (self.RNN.weight_hh_l0 ** 2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = tf.reduce_sum((pos - pred_pos) ** 2, -1)
        err = tf.math.sqrt(err)
        err = tf.reduce_mean(err)

        return loss, err

    def train_step(self, data):
        x, (pc_outputs, pos) = data
        with tf.GradientTape() as tape:
            # uncomment if bayesian:
            #losses = list()
            #for sample in range(self.n_samples):
            #    y_pred = self(x, training=True)  # Forward pass
            #    losses.append(self.compiled_loss(y, y_pred))
            #loss = tf.reduce_sum(self.losses) + tf.reduce_mean(tf.stack(losses))  # total KL + mean crossentropy
            loss, err = self.compute_loss(x, pc_outputs, pos)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        #self.compiled_metrics.update_state(y, y_pred) # uncomment and modify if reconstruction error can be treated as a metric
        return {m.name: m.result() for m in self.metrics}




    #priors and posteriors will be required later, for now let's operate on Dense
    def prior_standard(self, kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        prior_model = tf.keras.Sequential(
            [
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(n), scale_diag=tf.ones(n)),
                    convert_to_tensor_fn=tfp.distributions.Distribution.sample)
            ]
        )
        return prior_model

    def prior_initialized(self, weights, epsilon=0):
        def _prior(kernel_size, bias_size, dtype=None):
            n = kernel_size + bias_size
            prior_model = tf.keras.Sequential(
                [
                    tfp.layers.DistributionLambda(
                        lambda t: tfp.distributions.MultivariateNormalDiag(loc=weights[..., :n],
                                                                           scale_diag=weights[..., n:] + epsilon),
                        convert_to_tensor_fn=tfp.distributions.Distribution.sample
                    )
                ]
            )
            return prior_model

        return _prior

    def posterior_mean_field(self, kernel_size, bias_size, dtype=None, epsilon=0, init_stdev=np.sqrt(np.exp(-6.))):
        n = kernel_size + bias_size
        posterior_model = tf.keras.Sequential(
            [
                tfp.layers.VariableLayer(
                    2 * n, dtype, initializer=tfp.layers.BlockwiseInitializer(
                        [tf.keras.initializers.TruncatedNormal(stddev=0.1),
                         tf.keras.initializers.Constant(init_stdev)], sizes=[n, n])
                ),
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.MultivariateNormalDiag(loc=t[..., :n], scale_diag=t[..., n:] + epsilon),
                    convert_to_tensor_fn=tfp.distributions.Distribution.sample

                )
            ]
        )
        return posterior_model
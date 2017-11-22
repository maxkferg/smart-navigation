import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.contrib.rnn import BasicRNNCell
from tensorflow.contrib.rnn import static_rnn


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]

    def construct_rnn(self, obz, rnn_history_steps, rnn_hid_units, rnn_num_layers=1, reuse=False, name=""):
        """
        Generate an RNN that is applied to the last @rnn_history_steps
        @obs. An array with shape [batch_size, history, state_size]
        """
        print("Generating {0} RNN that is applied to {1} states".format(name,rnn_history_steps))

        x = tf.unstack(obz, num=rnn_history_steps, axis=1)

        # 1-layer Basic Cell with n_hidden units.
        cell = BasicRNNCell(rnn_hid_units,reuse=reuse)

        # generate prediction
        outputs, states = static_rnn(cell, x, dtype=tf.float32)

        # We only return the last output
        return outputs[-1]



class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        print("Creating: %s with reuse=%r"%(self.name,reuse))
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # RNN
            xo = self.construct_rnn(obs, rnn_history_steps=4, rnn_hid_units=64, rnn_num_layers=1, reuse=reuse, name=self.name)

            # Layer 1
            x = tf.layers.dense(xo, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            # Layer 2
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = x+xo

            # Output layer
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        print("Creating: %s with reuse=%r"%(self.name,reuse))
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # RNN
            xo = self.construct_rnn(obs, rnn_history_steps=4, rnn_hid_units=64, rnn_num_layers=1, reuse=reuse, name=self.name)

            # Layer 1
            x = tf.layers.dense(xo, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            # Layer 2
            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            x = x+xo

            # Output layer
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars

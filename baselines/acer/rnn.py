import numpy as np
import tensorflow as tf
import sonnet as snt
from tensorflow.contrib.rnn import BasicRNNCell
from tensorflow.contrib.rnn import static_rnn


class RNN(snt.AbstractModule):
    """
    to model  f(.|\phi_{\theta'}(x_i))
    parameters denoted by \theta'

    Input: the observed state, x: 1 x INPUT_DIM

    Output: a distribution over action ( a normal distribution with fixed diagonal co-variance)

    """
    def __init__(self, hidden_size, output_size, name="rnn_net"):
        """
        hidden_size = size of the neural net
        output size = dimensionality of the action space
        """
        super(RNN, self).__init__(name=name)
        self._history_steps = 4
        self._hidden_size = hidden_size
        self._name = name

    def _build(self, inputs):
        """Compute output Tensor from input Tensor."""
        x = tf.unstack(inputs, num=self._history_steps, axis=1)
        cell = BasicRNNCell(self._hidden_size, activation=tf.nn.relu)
        outputs, states = static_rnn(cell, x, dtype=tf.float32)
        return tf.concat([outputs[-1], inputs], axis=1)


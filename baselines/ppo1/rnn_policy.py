from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
import numpy as np
import gym.spaces
from baselines.common.distributions import make_pdtype
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import static_rnn


def resnet(inputs, hid_size, name):
    x = U.dense(inputs, hid_size, "%s_dense1"%name, weight_init=U.normc_initializer(1.0))
    #x = tf.contrib.layers.batch_norm(x)
    x = tf.nn.relu(x)
    x = U.dense(x, hid_size, "%s_dense2"%name, weight_init=U.normc_initializer(1.0))
    #x = tf.contrib.layers.batch_norm(x)
    x = tf.nn.relu(x+inputs)
    return x


class RnnPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, rnn_hid_units, gaussian_fixed_var=True):
        #assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        #with tf.variable_scope("obfilter"):
        #    self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        #obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        obz = ob

        # Apply rnn_to reduce history
        with tf.variable_scope("vffc"):
            state = self.rnn(obz, ob_space.shape[0], rnn_hid_units)
            for i in range(num_hid_layers):
                last_out = resnet(state, hid_size, "vffc%i"%(i+1))
            self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]

        # Apply rnn_to reduce history
        with tf.variable_scope("polfc"):
            state = self.rnn(obz, ob_space.shape[0], rnn_hid_units)
            for i in range(num_hid_layers):
                last_out = resnet(state, hid_size, "polfc%i"%(i+1))

            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = U.dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
            else:
                raise
                pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def rnn(self, obz, rnn_history_steps, rnn_hid_units, rnn_num_layers=1):
        """
        Generate an RNN that is applied to the last @rnn_history_steps
        @obs. An array with shape [batch_size, history, state_size]
        """
        print("Generating RNN that is applied to {0} states".format(rnn_history_steps))

        x = tf.unstack(obz, num=rnn_history_steps, axis=1)

        # 1-layer Basic Cell with n_hidden units.
        cell = GRUCell(rnn_hid_units)

        # generate prediction
        outputs, states = static_rnn(cell, x, dtype=tf.float32)

        # We only return the last output
        return outputs[-1] # Get the last tuple, and take the output component

    def value(self, stochastic, ob):
        """
        Compute the value.
        @ob must be (batch_size, observation_size).
        Return size (batch_size, )
        """
        ac1, vpred1 =  self._act(stochastic, ob)
        return vpred1

    def act(self, stochastic, ob):
        ob = ob[None] # Add the batch dimension
        ac1, vpred1 =  self._act(stochastic, ob)
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []


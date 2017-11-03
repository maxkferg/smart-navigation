from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
import numpy as np
import gym.spaces
from baselines.common.distributions import make_pdtype


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

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

        # Apply rnn_to reduce history
        obz = self.rnn(obz, ob_space.shape[0], rnn_hid_units)

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]

        last_out = obz
        for i in range(num_hid_layers):
            last_in = last_out
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            last_out = last_in+last_out

        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = U.dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        else:
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

        # 1-layer GRU with n_hidden units.
        cells = [tf.nn.rnn_cell.GRUCell(rnn_hid_units) for i in range(rnn_num_layers)]

        cells = tf.nn.rnn_cell.MultiRNNCell(cells)

        # generate prediction
        outputs, states = tf.nn.static_rnn(cells, x, dtype=tf.float32)

        # We only return the last output
        return outputs[-1]

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


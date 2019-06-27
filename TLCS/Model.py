import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf

class TwoModels:
    def __init__(self, num_states, num_actions, batch_size):
        # state sapce
        self._num_states = num_states

        # action space
        self._num_actions = num_actions

        self._batch_size = batch_size

        # input & output for policy network
        self._policy_states = None
        self._policy_logits = None
        self._policy_namescope = "policy"

        # placeholder for policy network
        self._target_states = None
        self._target_logits = None
        self._target_namescope = "target"

        self._optimizer = None
        self._var_init = None

        # every self._target_update_step many steps, we will copy weights from policy to target network
        self._target_update_step=10

        # now setup the two models
        self._define_policy_model()
        self._define_target_model()
        self._var_init = tf.global_variables_initializer()

    # policy network
    def _define_policy_model(self):
        # placeholders
        self._policy_states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._policy_q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)

        # list of nn layers
        fc1 = tf.layers.dense(self._policy_states, 400, activation=tf.nn.relu, name=self._policy_namescope + "0")
        fc2 = tf.layers.dense(fc1, 400, activation=tf.nn.relu, name=self._policy_namescope + "1")
        fc3 = tf.layers.dense(fc2, 400, activation=tf.nn.relu, name=self._policy_namescope + "2")
        fc4 = tf.layers.dense(fc3, 400, activation=tf.nn.relu, name=self._policy_namescope + "3")
        fc5 = tf.layers.dense(fc4, 400, activation=tf.nn.relu, name=self._policy_namescope + "4")
        self._policy_logits = tf.layers.dense(fc5, self._num_actions)

        # parameters
        loss = tf.losses.mean_squared_error(self._policy_q_s_a, self._policy_logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        #self._var_init = tf.global_variables_initializer()

    # taret network: same as policy network but holds different weights and do not have loss & optimizer
    def _define_target_model(self):
        # placeholders
        self._target_states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        #self._target_q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)

        # list of nn layers
        fc1 = tf.layers.dense(self._target_states, 400, activation=tf.nn.relu, name=self._target_namescope + "0")
        fc2 = tf.layers.dense(fc1, 400, activation=tf.nn.relu, name=self._target_namescope + "1")
        fc3 = tf.layers.dense(fc2, 400, activation=tf.nn.relu, name=self._target_namescope + "2")
        fc4 = tf.layers.dense(fc3, 400, activation=tf.nn.relu, name=self._target_namescope + "3")
        fc5 = tf.layers.dense(fc4, 400, activation=tf.nn.relu, name=self._target_namescope + "4")
        self._target_logits = tf.layers.dense(fc5, self._num_actions)

        # Todo: probably don't need this becaue we are not training the network
        # parameters
        #loss = tf.losses.mean_squared_error(self._target_q_s_a, self._logits)
        #self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        #self._var_init = tf.global_variables_initializer()

    # RETURNS THE OUTPUT OF THE NETWORK GIVEN A SINGLE STATE
    def policy_predict_one(self, state, sess):
        return sess.run(self._policy_logits, feed_dict={self._policy_states: state.reshape(1, self.num_states)})

    # RETURNS THE OUTPUT OF THE NETWORK GIVEN A BATCH OF STATES
    def policy_predict_batch(self, states, sess):
        return sess.run(self._policy_logits, feed_dict={self._policy_states: states})

    # TRAIN THE NETWORK
    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._policy_states: x_batch, self._policy_q_s_a: y_batch})

    def target_predict_batch(self, states, sess):
        return sess.run(self._target_logits, feed_dict={self._target_states: states})

    # Todo: implement copying weights from policy to target
    def copy_weights_to_target(self):
        for (new, old) in zip(tf.trainable_variables(self._policy_namescope),
                              tf.trainable_variables(self._target_namescope)):
            tf.assign(new, old)

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def var_init(self):
        return self._var_init

    @property
    def target_update_step(self):
        return self._target_update_step

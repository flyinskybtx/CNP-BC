# Model
import copy
import os
import os.path as osp

import gym
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.offline import JsonReader
from ray.rllib.offline.input_reader import _QueueRunner
from ray.rllib.utils import try_import_tf
import numpy as np

tf1, tf, version = try_import_tf()

from Data import DATA_DIR
from Data.cartpole_data import PolicyDataGenerator, CNPCartPoleData, PolicyData
from Models import MODEL_DIR


def logits_dist_loss(y_true, y_pred):
    """ categorical crossentropy for Discrete actions

    :param y_true:
    :param y_pred:
    :return:
    """
    losses = tf.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    return tf.reduce_mean(losses)


def logits_cate_acc(y_true, y_pred):
    """ Categorical accuracy for Discrete actions

    :param y_true:
    :param y_pred:
    :return:
    """
    accs = tf.metrics.sparse_categorical_accuracy(y_true, y_pred)
    return tf.reduce_mean(accs)


def build_fc_model(obs_space, action_space, hiddens, with_values=True, name='fc_model'):
    inputs = tf.keras.layers.Input(shape=obs_space.shape, name="obs")
    x = inputs
    for i, units in enumerate(hiddens[:-1], start=1):
        x = tf.keras.layers.Dense(units, name=f'dense_{i}', activation='relu',
                                  kernel_initializer=normc_initializer(1.0))(x)

    logits = tf.keras.layers.Dense(action_space.n, activation=None,
                                   kernel_initializer=normc_initializer(1.0),
                                   name="logits")(x)
    if with_values:
        values = tf.keras.layers.Dense(1, activation=None,
                                       kernel_initializer=normc_initializer(0.01), name="values")(x)
        outputs = [logits, values]
    else:
        outputs = [logits]
    return tf.keras.models.Model(inputs, outputs, name=name)


class PolicyFCModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """

        :param obs_space:
        :param action_space:
        :param num_outputs:
        :param model_config:
        :param name:
        """

        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        ## Create Model
        self.hiddens = model_config['custom_model_config']['hiddens']
        self.base_model = build_fc_model(obs_space, action_space, self.hiddens)
        self.register_variables(self.base_model.variables)
        self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        model_out, value_out = self.base_model(input_dict['obs'])
        self._value_out = value_out
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def import_from_h5(self, import_file):
        # Override this to define custom weight loading behavior from h5 files.
        self.base_model.load_weights(import_file, by_name=True)

    def custom_stats(self):
        print(f'Imitation_loss: {self.imitation_loss}')
        return {
            'policy_loss': self.policy_loss,
            'imitation_loss': self.imitation_loss,
        }

    def supervised_pretrain(self, name, batch_size=16):
        """  Use surrogate model to do supervised learning and save model weights
        :param name:
        :param batch_size:
        :return:
        """
        # Create surrogate models, use

        surrogate_model = build_fc_model(self.obs_space, self.action_space, self.hiddens, with_values=False)
        surrogate_file = osp.join(DATA_DIR, f'intermediate/{name}.h5')

        # load previous trained model
        if os.path.exists(surrogate_file):
            surrogate_model.load_weights(surrogate_file, by_name=True)

        surrogate_model.compile(
            loss={'logits': logits_dist_loss},
            metrics={'logits': logits_cate_acc},
            optimizer=tf.keras.optimizers.Adam(lr=5e-4))

        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            action_dims = 1
        else:  # MultiDiscretes
            action_dims = self.action_space.nvec

        train_data = PolicyDataGenerator(self.model_config['custom_model_config']['offline_dataset'],
                                         batch_size=batch_size,
                                         action_dims=action_dims)
        vali_data = copy.deepcopy(train_data)

        hist = surrogate_model.fit(
            train_data, epochs=5, steps_per_epoch=100, verbose=1,
            validation_data=vali_data, validation_steps=20,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, mode='auto', restore_best_weights=True),
            ]
        )

        surrogate_model.save_weights(surrogate_file)
        print(f'Save surrogate weights to file {surrogate_file}')
        print(
            f'Train results: loss:{hist.history["loss"][-1]}, accuracy: {hist.history["logits_cate_acc"][-1]}')


class SurrogateFCModel:
    def __init__(self, obs_space, action_space, hiddens, offline_dataset, name):
        self.obs_space = obs_space
        self.action_space = action_space
        self.hiddens = hiddens
        self.offline_dataset = offline_dataset
        self.name = f'surrogate_{name}'
        self.weights_file = osp.join(DATA_DIR, f'intermediate/{name}.h5')

        self.base_model = build_fc_model(obs_space, action_space, hiddens, with_values=False)
        self.base_model.compile(
            loss={'logits': logits_dist_loss},
            metrics={'logits': logits_cate_acc},
            optimizer=tf.keras.optimizers.Adam(lr=5e-4)
        )

    def train(self, epochs=5, batch_size=16, verbose=1):
        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            action_dims = 1
        else:  # MultiDiscretes
            action_dims = self.action_space.nvec

        train_data = PolicyDataGenerator(self.offline_dataset,
                                         batch_size=batch_size,
                                         action_dims=action_dims)
        vali_data = copy.deepcopy(train_data)

        hist = self.base_model.fit(
            train_data, epochs=epochs, steps_per_epoch=100, verbose=verbose,
            validation_data=vali_data, validation_steps=20,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=3, mode='auto', restore_best_weights=True),
            ]
        )
        print(f'Train Surrogate model: loss:{hist.history["loss"][-1]}, accuracy:'
              f' {hist.history["logits_cate_acc"][-1]}')

    def load_weights(self, weights_file=None):
        if weights_file is None:
            # load previous trained model
            self.base_model.load_weights(self.weights_file, by_name=True)
        else:
            self.base_model.load_weights(weights_file, by_name=True)

    def save_weights(self):
        self.base_model.save_weights(self.weights_file)


class PolicyCustomLossModel(PolicyFCModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        if self.model_config['custom_model_config']['offline_dataset'] is not None:
            self.data_generator = PolicyData(savedir=self.model_config['custom_model_config']['offline_dataset'],
                                             batch_size=16, )

    def update_data_generator(self):
        self.data_generator.load_all()
        self.data_generator.shuffle()

    def custom_loss(self, policy_loss, loss_inputs):
        if self.model_config['custom_model_config']['offline_dataset'] is not None:
            batch = self.data_generator.next()
            keys = [
                k for k in sorted(batch.keys())
                if np.issubdtype(batch[k].dtype, np.number)
            ]
            dtypes = [batch[k].dtype for k in keys]
            shapes = {
                k: (-1,) + s[1:]
                for (k, s) in [(k, batch[k].shape) for k in keys]
            }
            queue = tf1.FIFOQueue(capacity=1, dtypes=dtypes, names=keys)
            tensors = queue.dequeue()

            self._queue_runner = _QueueRunner(self.data_generator, queue, keys, dtypes)
            self._queue_runner.enqueue(batch)
            self._queue_runner.start()
            out = {k: tf.reshape(t, shapes[k]) for k, t in tensors.items()}

            obs = out['obs']
            actions = out['actions']
            num_samples = tf.shape(obs)[0]
            obs = tf.repeat(obs, repeats=tf.cast(tf.math.ceil(200 / num_samples), tf.int8), axis=0)[:200]
            actions = tf.repeat(actions, repeats=tf.cast(tf.math.ceil(200 / num_samples), tf.int8), axis=0)[:200]

            logits, _ = self.forward({'obs': obs}, [], None)
            action_dist = Categorical(logits, self.model_config)

            logp = -action_dist.logp(actions)

            self.imitation_loss = tf.reduce_mean(logp)
        else:
            self.imitation_loss = 0
        self.policy_loss = policy_loss
        print(f'Imitation_loss: {self.imitation_loss}')

        return policy_loss + 0.5 * self.imitation_loss

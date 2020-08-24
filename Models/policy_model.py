# Model
import copy
import os.path as osp

import gym
import tensorflow as tf
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.offline import JsonReader
from tensorflow import keras

from Data import DATA_DIR
from Data.cartpole_data import PolicyDataGenerator
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
        inputs = keras.layers.Input(shape=obs_space.shape, name="obs")
        x = inputs
        hiddens = model_config['custom_model_config']['hiddens']
        for i, units in enumerate(hiddens[:-1], start=1):
            x = keras.layers.Dense(units, name=f'dense_{i}', activation='relu',
                                   kernel_initializer=normc_initializer(1.0))(x)

        logits = keras.layers.Dense(self.num_outputs, activation=None,
                                    kernel_initializer=normc_initializer(1.0),
                                    name="logits")(x)
        values = keras.layers.Dense(1, activation=None,
                                    kernel_initializer=normc_initializer(0.01), name="values")(x)

        ## Create Model
        self.base_model = keras.Model(inputs=inputs,
                                      outputs=[logits, values])
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
        self.base_model.load_weights(import_file)

    def custom_loss(self, policy_loss, loss_inputs):
        if self.model_config['custom_model_config']['offline_dataset'] is not None:
            savedir = osp.join(DATA_DIR, self.model_config['custom_model_config']['offline_dataset'])
            reader = JsonReader(savedir)
            input_ops = reader.tf_input_ops()

            obs = input_ops['obs']
            actions = input_ops['actions']
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

        return policy_loss + 0.5 * self.imitation_loss

    def custom_stats(self):
        return {
            'policy_loss': self.policy_loss,
            'imitation_loss': self.imitation_loss,
        }

    def initialization(self, train=False, batch_size=16):
        if train:
            self.base_model.compile(
                loss={'logits': logits_dist_loss},
                metrics={'logits': logits_cate_acc},
                optimizer=keras.optimizers.Adam(lr=5e-4))

            if isinstance(self.action_space, gym.spaces.discrete.Discrete):
                action_dims = 1
            else:  # MultiDiscretes
                action_dims = self.action_space.nvec

            train_data = PolicyDataGenerator(self.model_config['custom_model_config']['offline_dataset'],
                                             batch_size=batch_size,
                                             action_dims=action_dims)
            vali_data = copy.deepcopy(train_data)

            hist = self.base_model.fit(
                train_data, epochs=100, steps_per_epoch=100, verbose=0,
                validation_data=vali_data, validation_steps=20,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=5, mode='auto', restore_best_weights=True),
                ]
            )
            print(
                f'Train results: loss:{hist.history["loss"][-1]}, accuracy: {hist.history["logits_logits_cate_acc"][-1]}')
            self.base_model.save_weights(osp.join(MODEL_DIR, f'Checkpoints/{self.name}_supervised.h5'))

        else:
            self.base_model.load_weights(osp.join(MODEL_DIR, f'Checkpoints/{self.name}_supervised.h5'))

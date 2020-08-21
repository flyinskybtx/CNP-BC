# Model
from typing import Dict

import tensorflow as tf
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.offline import JsonReader
from ray.rllib.utils.types import TensorType
from tensorflow import keras


class FCModel(TFModelV2):
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
            x = keras.layers.Dense(units, name=f'dense_{i}', activation='tanh')(x)

        logits = keras.layers.Dense(self.num_outputs, activation='linear', name="logits")(x)
        values = keras.layers.Dense(1, activation=None, name="values")(x)

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
        reader = JsonReader(self.model_config['custom_model_config']['offline_dataset'])
        input_ops = reader.tf_input_ops()

        supervised_obs = input_ops['obs']
        logits, _ = self.forward({'obs': supervised_obs}, [], None)
        action_dist = Categorical(logits, self.model_config)

        logp = -action_dist.logp(input_ops['actions'])

        self.imitation_loss = tf.reduce_mean(logp)
        self.policy_loss = policy_loss

        return policy_loss + self.imitation_loss

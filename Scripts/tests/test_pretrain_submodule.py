import copy
import os.path as osp

import gym
import numpy as np
import ray.rllib
import tensorflow as tf

from Data import DATA_DIR
from Data.cartpole_data import PolicyDataGenerator
from Models import MODEL_DIR
from Models.policy_model import PolicyFCModel, logits_dist_loss


def train(obs_space, action_space, num_outputs, model_config, name):
    # TODO: create a single output model, save weights then load weights by name
    model = PolicyFCModel(obs_space,
                          action_space,
                          num_outputs,
                          model_config,
                          name)
    model.base_model.summary()

    surrogant_model = tf.keras.Model(model.base_model.input, model.base_model.outputs[0])
    train_data = PolicyDataGenerator(model.model_config['custom_model_config']['offline_dataset'],
                                     batch_size=16,
                                     action_dims=1)
    vali_data = copy.deepcopy(train_data)
    surrogant_model.compile(loss={'logits':logits_dist_loss})

    records = surrogant_model.fit(
        train_data, epochs=5, steps_per_epoch=100, validation_data=vali_data, validation_steps=20,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, mode='auto', restore_best_weights=True),
        ]
    )

    surrogant_model.save_weights(osp.join(DATA_DIR, f'intermediate/surrogate_{model.base_model.name}.h5'))
    model.base_model.load_weights(osp.join(DATA_DIR, f'intermediate/surrogate_{model.base_model.name}.h5'),
                                  by_name=True)


if __name__ == '__main__':
    train(gym.spaces.Box(0, 1, (4,)), gym.spaces.Discrete(2), num_outputs=2, model_config={
        'custom_model': 'dummy_model',
        "custom_model_config": {
            'hiddens': [256, 128, 16],
            'offline_dataset': f'offline/cartpole/Naive_MPC',
        },
    }, name='dummy')

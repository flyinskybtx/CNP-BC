import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os.path as osp

from Data import DATA_DIR
from Data.cartpole_data import PolicyDataGenerator
from Envs.custom_cartpole_v1 import CustomCartPole
from Models import MODEL_DIR
from Models.policy_model import PolicyFCModel, logits_dist_loss, logits_cate_acc

if __name__ == '__main__':
    env_configs = {'masscart': 1.0,
                   'masspole': 0.1,
                   'length': 0.5,
                   'force_mag': 10, }
    env = CustomCartPole(env_configs)

    model_config = {
        'custom_model': 'dummy_supervised',
        "custom_model_config": {
            'hiddens': [256, 128, 16],
            'offline_dataset': f'offline/cartpole/Naive_MPC',
        }
    }

    model = PolicyFCModel(env.observation_space, env.action_space, 2, model_config, 'dummy_supervised')
    model.base_model.summary()

    model.base_model.compile(
        loss={'logits': logits_dist_loss},
        metrics={'logits': logits_cate_acc},
        optimizer=keras.optimizers.Adam(lr=5e-3))

    dataset = PolicyDataGenerator(model_config['custom_model_config']['offline_dataset'], batch_size=16, action_dims=1)
    vali_dataset = copy.deepcopy(dataset)

    records = model.base_model.fit(
        dataset, epochs=5, steps_per_epoch=100, validation_data=vali_dataset, validation_steps=20,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, mode='auto', restore_best_weights=True),
        ]
    )
    model.base_model.save_weights(osp.join(MODEL_DIR, f'Checkpoints/Naive_MPC.h5'))
    xs, ys, = dataset.__getitem__(0)
    logits, values = model.base_model.predict(xs)
    y_preds = np.argmax(logits, axis=-1).reshape(-1, 1)
    y_true = ys.reshape(-1, 1)
    print(np.concatenate([y_true, y_preds], axis=-1))
    print(records.history)

import random
import string

import numpy as np
import ray
from ray.rllib.agents import a3c
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf
from ray.tune import register_env

from Data.cartpole_data import PolicyDataGenerator

_, tf, _ = try_import_tf()

from Envs.custom_cartpole_v1 import CustomCartPole
from Models.policy_model import PolicyFCModel, logits_dist_loss, logits_cate_acc
from Scripts.tests.test_pretrain_submodule import train

if __name__ == '__main__':
    ray.shutdown(True)
    ray.init(num_gpus=1)
    register_env('BaselineCartPole-v1', lambda config: CustomCartPole(config))
    #
    model_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    ModelCatalog.register_custom_model(model_name, PolicyFCModel)

    print('------------------------------')
    rl_config = {
        "train_batch_size": 200,
        'num_workers': 0,
        'env_config': {'masscart': 1.0,
                       'masspole': 0.1,
                       'length': np.random.uniform(low=0.5, high=1.0),
                       'force_mag': 10, },
        'model': {
            'custom_model': model_name,
            "custom_model_config": {
                'hiddens': [256, 128, 16],
                'offline_dataset': f'offline/cartpole/Naive_MPC',
            },
        },
    }
    env = CustomCartPole(rl_config['env_config'])

    agent = a3c.A2CTrainer(config=rl_config, env='BaselineCartPole-v1')

    # Test get weights from agent
    def get_weights_from_agent(agent):
        weights = agent.get_weights()["default_policy"]
        return weights
    print(get_weights_from_agent(agent))

    # Test save policy as h5 file
    with agent.get_policy()._sess.graph.as_default():
        with agent.get_policy()._sess.as_default():
            agent.get_policy().model.base_model.save_weights("./policy_weights.h5")
    print("SAVE: after save", get_weights_from_agent(agent))


    # Test load policy from h5 file
    agent.workers.local_worker().get_policy().import_model_from_h5("./policy_weights.h5")
    agent.workers.sync_weights()


    # Test Train directly
    with agent.get_policy()._sess.graph.as_default():
        with agent.get_policy()._sess.as_default():
            model = agent.get_policy().model
            model.base_model.compile(
                loss={'logits': logits_dist_loss},
                metrics={'logits': logits_cate_acc},
                optimizer=tf.keras.optimizers.Adam(lr=5e-4))
            train_data = PolicyDataGenerator(model.model_config['custom_model_config']['offline_dataset'],
                                             batch_size=16,
                                             action_dims=1)
            model.base_model.fit(train_data, epochs=10)

import random
import string

import numpy as np
import ray
from ray.rllib import rollout
from ray.rllib.agents import dqn
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from tqdm import tqdm

from Algorithms.mpc import CemMPCController
from Data.basics import rollout_and_save_data, remove_data, gen_context
from Data.cartpole_data import CNPCartPoleGenerator
from Envs.custom_cartpole_v1 import CustomCartPole, make_cartpole_reward_on_traj
from Models.cnp_model import CNPModel
from Models.policy_model import FCModel

if __name__ == '__main__':
    ray.shutdown(True)
    ray.init(num_gpus=1)
    register_env('BaselineCartPole-v1', lambda config: CustomCartPole(config))

    model_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    ModelCatalog.register_custom_model(model_name, FCModel)

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
                'offline_dataset': None,
            },
        },
    }
    results = []
    dqn_trainer = dqn.DQNTrainer(config=rl_config, env='BaselineCartPole-v1')

    for i in tqdm(range(100)):
        result_mf = dqn_trainer.train()
        print(f"\t RL Reward: "
              f"{result_mf['episode_reward_max']:.4f}  |  "
              f"{result_mf['episode_reward_mean']:.4f}  |  "
              f"{result_mf['episode_reward_min']:.4f}")
        results.append(result_mf['episode_reward_mean'])
        if i % 20 == 0:
            checkpoint = dqn_trainer.save()
            print("checkpoint saved at", checkpoint)
        if i % 20 == 0:
            rollout.rollout(dqn_trainer, env_name='CustomCartPole-v1', num_steps=50, num_episodes=1, no_render=False)

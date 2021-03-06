import random
import string
import numpy as np

import ray
from ray.rllib import rollout
from ray.rllib.agents import pg
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from tqdm import tqdm

from Envs.custom_cartpole_v1 import CustomCartPole
from Models.policy_model import PolicyFCModel

if __name__ == '__main__':

    ray.shutdown(True)
    ray.init(num_gpus=1, )
    register_env('CustomCartPole-v1', lambda config: CustomCartPole(config))

    # Model Free
    model_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    ModelCatalog.register_custom_model(model_name, PolicyFCModel)
    config_rl = {
        "train_batch_size": 200,
        'num_workers': 0,
        'log_level': 'INFO',
        'framework': 'tf',
        'env_config': {
            'masscart': 1.0,
            'masspole': 0.1,
            'length': np.random.uniform(0.5, 1),
            'force_mag': 10,
        },
        'model': {
            'custom_model': model_name,
            "custom_model_config": {
                'hiddens': [32, 32, 16],
                'offline_dataset': 'offline/cartpole/Cem_MPC'
            },
        },
    }

    results = {'mf': [], 'mpc-bc': []}
    agent = pg.PGTrainer(config=config_rl, env='CustomCartPole-v1')

    for i in tqdm(range(100)):
        result_mf = agent.train()

        print(f"\t RL Reward: "
              f"{result_mf['episode_reward_max']:.4f}  |  "
              f"{result_mf['episode_reward_mean']:.4f}  |  "
              f"{result_mf['episode_reward_min']:.4f}  |"
              )

        results['mf'].append(result_mf['episode_reward_mean'])

        if i % 50 == 0:
            checkpoint = agent.save()
            print("checkpoint saved at", checkpoint)

        if i % 50 == 0:
            rollout.rollout(agent, env_name='CustomCartPole-v1', num_steps=50, num_episodes=1, no_render=False)

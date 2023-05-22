import random
import string

import numpy as np
import ray
from ray.rllib.algorithms.pg import PGConfig
from ray.tune import register_env
from ray.tune.logger import pretty_print

from Envs.custom_cartpole_v1 import CustomCartPole

# from Models.policy_model import PolicyFCModel

if __name__ == '__main__':
    ray.init(num_gpus=1)
    register_env('BaselineCartPole-v1', lambda config: CustomCartPole(config))
    
    model_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    # ModelCatalog.register_custom_model(model_name, PolicyFCModel)
    
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
    algo_config = PGConfig()
    algo_config.rollouts(num_rollout_workers=1)
    algo_config.resources(num_gpus=0)
    algo_config.environment(
        env='BaselineCartPole-v1',
        env_config={'masscart': 1.0,
                    'masspole': 0.1,
                    'length': np.random.uniform(low=0.5, high=1.0),
                    'force_mag': 10, },
    )
    algo = algo_config.build()
    for i in range(10):
        result = algo.train()
        print(pretty_print(result))
        
        if i % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
    
    # agent = pg.PGTrainer(config=rl_config, env='BaselineCartPole-v1')
    # print(agent.get_policy().model.base_model.summary())
    #
    # for i in tqdm(range(200)):
    #     result_mf = agent.train_model()
    #     # if i % 20 == 0:
    #     #     checkpoint = agent.save()
    #     #     print("checkpoint saved at", checkpoint)
    #     if i % 20 == 0:
    #         print(f"\t RL Reward: "
    #               f"{result_mf['episode_reward_max']:.4f}  |  "
    #               f"{result_mf['episode_reward_mean']:.4f}  |  "
    #               f"{result_mf['episode_reward_min']:.4f}")
    #         results.append(result_mf['episode_reward_mean'])
    #         rollout.rollout(agent, env_name='CustomCartPole-v1', num_steps=200, num_episodes=1, no_render=True)

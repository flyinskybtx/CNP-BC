import numpy as np
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env
import os.path as osp

from Data import WEIGHTS_DIR
from Envs.custom_cartpole_v1 import CustomCartPole

if __name__ == '__main__':
    env_name = 'BaselineCartPole-v1'
    env_config = {'masscart': 1.0,
                  'masspole': 0.1,
                  'length': np.random.uniform(low=0.5, high=1.0),
                  'force_mag': 10, }
    register_env(env_name, lambda config: CustomCartPole(config))
    
    env = CustomCartPole(env_config)
    algo = DQNConfig().environment(env=env_name, env_config=env_config).build()
    
    policy = algo.get_policy()
    weights = algo.get_policy().get_weights()
    weights_file = osp.join(WEIGHTS_DIR, 'test_access_policy_state.h5')
    print(weights)
        

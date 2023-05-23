import numpy as np
import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune import register_env

from Envs.custom_cartpole_v1 import CustomCartPole

if __name__ == '__main__':
    ray.init()
    
    env_name = 'BaselineCartPole-v1'
    env_config = {'masscart': 1.0,
                  'masspole': 0.1,
                  'length': np.random.uniform(low=0.5, high=1.0),
                  'force_mag': 10, }
    register_env(env_name, lambda config: CustomCartPole(config))
    
    env = CustomCartPole(env_config)
    algo = DQNConfig().environment(env=env_name, env_config=env_config).framework("tf2").build()
    
    policy = algo.get_policy()
    
    obs, info = env.reset()
    logits, _ = policy.model({'obs': np.expand_dims(obs, 0)})  # !! 注意这里应该是输入一个batch，而不是单独的一个obs

    print(logits)
    
    dist = policy.dist_class(logits, policy.model)
    print(dist.sample())
    print(dist.logp([1]))
    print(policy.model.value_function())
    policy.model.base_model.summary()

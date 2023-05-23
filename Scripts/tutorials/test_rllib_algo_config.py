import numpy as np
import ray
from ray import tune, air
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.callbacks import MemoryTrackingCallbacks
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
    
    alg_config = AlgorithmConfig()
    alg_config.training(gamma=0.9, lr=0.01)
    alg_config.framework(framework='tf2', eager_tracing=True)
    alg_config.environment(env=env_name, env_config=env_config)
    alg_config.resources(num_gpus=0)
    alg_config.rollouts(num_rollout_workers=4)
    alg_config.callbacks(MemoryTrackingCallbacks)
    alg_config.reporting(min_train_timesteps_per_iteration=10)
    alg_config.checkpointing(export_native_model_files=True)
    alg_config.debugging(log_level='INFO', log_sys_usage=True, )
    
    # 用Tuner
    alg_config.training(lr=tune.grid_search([0.01, 0.001]))
    results = tune.Tuner(
        "DQN",
        param_space=alg_config.to_dict(),
    ).fit()
    
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")

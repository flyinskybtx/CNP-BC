import numpy as np
import ray
from ray import tune, air
from ray.rllib.algorithms import AlgorithmConfig
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
    alg_config.framework(framework='tf2', eager_tracing=False)
    alg_config.environment(env=env_name, env_config=env_config)
    alg_config.resources(num_gpus=0)
    alg_config.checkpointing(export_native_model_files=True)
    alg_config.debugging(log_level='INFO', log_sys_usage=True, )
    alg_config.training(lr=0.001)
    
    # Run Config
    run_config = air.RunConfig(
        stop={'episode_reward_mean': 80},
        checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True),
    )
    results = tune.Tuner(
        "DQN",
        param_space=alg_config.to_dict(),
        run_config=run_config,
    ).fit()
    
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    best_checkpoint = best_result.checkpoint
    print(best_result)

import numpy as np
from ray.rllib.algorithms.ars import ARSConfig, ARSTFPolicy
from ray.rllib.algorithms.dqn import DQNTFPolicy
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.tune import register_env

from Envs.custom_cartpole_v1 import CustomCartPole

if __name__ == '__main__':
    # Register Env
    env_name = 'BaselineCartPole-v1'
    register_env(env_name, lambda config: CustomCartPole(config))
    
    env_config = {'masscart': 1.0,
                  'masspole': 0.1,
                  'length': np.random.uniform(low=0.5, high=1.0),
                  'force_mag': 10, }
    env = CustomCartPole(env_config)
    
    # 需要一个策略生成随机动作
    
    # 创造Config
    algo_config = ARSConfig()
    algo_config.framework(framework='tf2', eager_tracing=False)
    algo_config.training(sgd_stepsize=0)
    algo_config = algo_config.resources(num_gpus=0)
    algo_config.environment(env=env_name, env_config=env_config)
    algo = algo_config.build()
    
    # 创造worker
    workers = WorkerSet(
        # default_policy_class=policy.__class__,
        default_policy_class=ARSTFPolicy,
        env_creator=lambda _config: CustomCartPole(_config),
        config=algo_config,
        num_workers=4)
    
    while True:
        # 收集Batch样本
        samples = workers.foreach_worker(func=lambda w: w.sample()['default_policy'], local_worker=False,
                                         healthy_only=True)
        print(samples)
        # todo: 处理样本
        # todo: 储存样本

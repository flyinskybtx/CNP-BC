import numpy as np
import ray
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.dqn import DQN, DQNConfig, DQNTFPolicy
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import concat_samples
from ray.tune import register_env

from Envs.custom_cartpole_v1 import CustomCartPole

if __name__ == '__main__':
    # Setup policy and rollout workers.
    env_name = 'BaselineCartPole-v1'
    env_config = {'masscart': 1.0,
                  'masspole': 0.1,
                  'length': np.random.uniform(low=0.5, high=1.0),
                  'force_mag': 10, }
    register_env(env_name, lambda config: CustomCartPole(config))
    env = CustomCartPole(env_config)
    
    alg_config = DQNConfig()
    alg_config.framework(framework='tf2', eager_tracing=True)
    alg_config.environment(env=env_name, env_config=env_config)
    alg_config.resources(num_gpus=0)
    alg_config.checkpointing(export_native_model_files=True)
    alg_config.debugging(log_level='INFO', log_sys_usage=True, )
    alg_config.training(lr=0.001)
    algo = alg_config.build()
    policy = algo.get_policy()
    
    print(policy.__class__)
    
    workers = WorkerSet(
        # default_policy_class=policy.__class__,
        default_policy_class=DQNTFPolicy,
        env_creator=lambda _config: CustomCartPole(_config),
        config=alg_config,
        num_workers=2)
    
    while True:
        # Gather a batch of samples.
        samples = workers.foreach_worker(func=lambda w: w.sample()['default_policy'], local_worker=False, healthy_only=True)
        print(samples)
        T1 = concat_samples(samples)
        
        # Improve the policy using the T1 batch.
        policy.learn_on_batch(T1)
        
        # The local worker acts as a "parameter server" here.
        weights = policy.get_weights()
        workers.foreach_worker(func=lambda w: w.set_weights({"default_policy": weights}), local_worker=False)
        print("finish")

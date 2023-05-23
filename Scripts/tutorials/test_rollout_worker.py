import numpy as np
import ray
from ray.rllib import SampleBatch
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.examples.rollout_worker_custom_workflow import CustomPolicy
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
    
    algo = DQNConfig().environment(env=env_name, env_config=env_config).build()
    policy = algo.get_policy()

    workers = WorkerSet(
        policy_class=policy.__class__,
        env_creator=lambda _config: CustomCartPole(_config),
        num_workers=10)
    
    while True:
        # Gather a batch of samples.
        T1 = SampleBatch.concat_samples(
            ray.get([w.sample.remote() for w in workers.remote_workers()]))
        
        # Improve the policy using the T1 batch.
        policy.learn_on_batch(T1)
        
        # The local worker acts as a "parameter server" here.
        # We put the weights of its `policy` into the Ray object store once (`ray.put`)...
        weights = ray.put({"default_policy": policy.get_weights()})
        for w in workers.remote_workers():
            # ... so that we can broacast these weights to all rollout-workers once.
            w.set_weights.remote(weights)

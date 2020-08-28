import random
import string
import os.path as osp
import ray
from ray.rllib import rollout
from ray.rllib.agents import dqn, a3c
from ray.rllib.evaluation import SampleBatchBuilder
from ray.rllib.models import ModelCatalog
from ray.rllib.offline import JsonWriter
from ray.rllib.rollout import RolloutSaver
from ray.tune import register_env
from tqdm import tqdm
import numpy as np

from Data import DATA_DIR
from Envs.custom_cartpole_v1 import CustomCartPole
from Models.policy_model import PolicyFCModel

if __name__ == '__main__':

    ray.shutdown(True)
    ray.init(num_gpus=1)
    register_env('BaselineCartPole-v1', lambda config: CustomCartPole(config))

    model_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    ModelCatalog.register_custom_model(model_name, PolicyFCModel)

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
    agent = a3c.A3CTrainer(config=rl_config, env='BaselineCartPole-v1')
    saver = RolloutSaver(outfile=osp.join(DATA_DIR, f'rollout/dqn'))

    savedir = osp.join(DATA_DIR, f'offline/cartpole/rollout')

    batch_builder = SampleBatchBuilder()
    writer = JsonWriter(f"{savedir}")

    # ---------------------------------------------------

    for i in tqdm(range(10)):
        rollout.rollout(agent, env_name='CustomCartPole-v1',
                        num_steps=13, num_episodes=1, no_render=True,
                        saver=saver)
        saver.end_rollout()
        for data in saver._rollouts[0]:
            obs, action, new_obs, reward, done = data

            # -------------------------------------------------------
            batch_builder.add_values(
                obs=obs,
                new_obs=new_obs,
                actions=action,
                dones=done,
                reward=reward,
                infos=rl_config['env_config']
            )
        writer.write(batch_builder.build_and_reset())

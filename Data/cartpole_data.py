import glob
import os
import os.path as osp
from collections import defaultdict

import numpy as np
from ray.rllib.evaluation import SampleBatchBuilder
from ray.rllib.offline import JsonWriter, JsonReader
from tensorflow import keras
from tqdm import tqdm

from Envs.custom_cartpole_v1 import CustomCartPole


class CartPoleGenerator(keras.utils.Sequence):
    def __init__(self, savedir, batch_size, num_context, train=False):
        self.savedir = osp.abspath(osp.join(osp.dirname(__file__), savedir))
        self.reader = JsonReader(self.savedir)
        self.batch_size = batch_size
        self.min_num_context, self.max_num_context = num_context
        self.train = train
        self.on_epoch_end()

    def __getitem__(self, index):
        batch = self.get_batch(self.batch_size + self.num_context)
        obs = batch['obs']
        new_obs = batch['new_obs']
        actions = batch['actions'].reshape(-1, 1)
        states = np.concatenate([obs, actions], axis=-1)
        delta = new_obs - obs

        idx = np.arange(obs.shape[0])
        np.random.shuffle(idx)

        if self.train:
            context_x = np.repeat(np.expand_dims(states[idx[:self.num_context]], axis=0),
                                  states.shape[0], axis=0)
            context_y = np.repeat(np.expand_dims(delta[idx[:self.num_context]], axis=0),
                                  states.shape[0], axis=0)
            query_x = states  # If train, include context points
            target_y = delta
        else:
            context_x = np.repeat(np.expand_dims(states[idx[:self.num_context]], axis=0),
                                  self.batch_size, axis=0)
            context_y = np.repeat(np.expand_dims(delta[idx[:self.num_context]], axis=0),
                                  self.batch_size, axis=0)
            query_x = states[idx[self.num_context:]]  # If test, select other points as query
            target_y = delta[idx[self.num_context:]]

        return {'context_x': context_x, 'context_y': context_y, 'query_x': query_x}, target_y

    def get_batch(self, batch_size):
        batch = defaultdict(list)
        sample = self.reader.next()
        for k, v in sample.items():
            batch[k].append(v)

        chosen_length = sample['length'][0]
        while len(np.concatenate(batch['length'])) < batch_size:
            sample = self.reader.next()
            if sample['length'][0] == chosen_length:
                for k, v in sample.items():
                    batch[k].append(v)
        return {k: np.concatenate(v)[:batch_size] for k, v in batch.items()}

    def on_epoch_end(self):
        self.num_context = np.random.randint(self.min_num_context, self.max_num_context)

    def __len__(self):
        return 1000000

    def remove_data(self, pattern='random/*.json'):
        """ Remove old data according to pattern in 'savedir'

        :param pattern:
        :return:
        """
        old_files = glob.glob(f"{self.savedir}/{pattern}")
        for f in old_files:
            os.remove(f)
            print(f'Removed file: {f}')


def rollout_and_save_data(savedir, num_configs, episodes, steps_per_episode, controller=None):
    savedir = osp.abspath(osp.join(osp.dirname(__file__), savedir))
    batch_builder = SampleBatchBuilder()
    writer = JsonWriter(f"{savedir}")

    pbar = tqdm(total=num_configs * episodes)
    for _ in range(num_configs):
        env_config = {
            'masscart': 1.0,
            'masspole': 0.1,
            'length': np.random.uniform(0.5, 1),
            'force_mag': 10,
        }
        env = CustomCartPole(env_config)
        for eps_id in range(episodes):
            obs = env.reset()
            done = False
            t = 0

            while t < steps_per_episode:
                if done:
                    break
                if controller is None:
                    action = env.action_space.sample()
                else:
                    action, _ = controller.next_action(obs)

                new_obs, rew, done, info = env.step(int(action))
                batch_builder.add_values(
                    length=env_config["length"],
                    t=t,
                    eps_id=eps_id,
                    obs=obs,
                    new_obs=new_obs,
                    actions=action,
                    dones=done,
                    infos={},
                )
                obs = new_obs.copy()
                t += 1
            writer.write(batch_builder.build_and_reset())
            pbar.update(1)


if __name__ == '__main__':
    # cartpole_data_gen.remove_data('*.json')
    rollout_and_save_data(num_configs=1, episodes=10000, steps_per_episode=1000, savedir='offline/cartpole/Cem_MPC')
    cartpole_data_gen = CartPoleGenerator(savedir='offline/cartpole/random', batch_size=16, num_context=[10, 20])
    print(cartpole_data_gen.__getitem__(0)[0]['context_y'])

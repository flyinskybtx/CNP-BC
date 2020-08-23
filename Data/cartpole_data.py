import os.path as osp
from collections import defaultdict

import os.path as osp
from collections import defaultdict

import numpy as np
from ray.rllib.offline import JsonReader
from tensorflow import keras


class CNPCartPoleGenerator(keras.utils.Sequence):
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

        chosen_length = sample['infos'][0]['length']
        while len(batch['t']) < batch_size:
            sample = self.reader.next()
            if sample['infos'][0]['length'] == chosen_length:
                for k, v in sample.items():
                    batch[k].append(v)
        return {k: np.concatenate(v)[:batch_size] for k, v in batch.items()}

    def on_epoch_end(self):
        self.num_context = np.random.randint(self.min_num_context, self.max_num_context)

    def __len__(self):
        return 1000000


if __name__ == '__main__':
    cartpole_data_gen = CNPCartPoleGenerator(savedir='offline/cartpole/random',
                                             batch_size=16,
                                             num_context=[10, 20],
                                             train=True)
    print(cartpole_data_gen.__getitem__(0)[0]['context_y'])

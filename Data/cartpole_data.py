import glob
import os.path as osp
import random
from collections import defaultdict

import os.path as osp
from collections import defaultdict

import numpy as np
from ray.rllib.offline.json_reader import _from_json
from tensorflow import keras
from ray.rllib.offline import JsonReader
import itertools

from Data import DATA_DIR


class CNPCartPoleGenerator(keras.utils.Sequence):
    def __init__(self, savedir, batch_size, num_context, action_dims=1, train=False):
        """

        :param savedir:
        :param batch_size:
        :param num_context:
        :param action_dims:
        :param train:
        """
        self.savedir = osp.join(DATA_DIR, savedir)

        self.reader = JsonReader(self.savedir)
        self.batch_size = batch_size
        self.min_num_context, self.max_num_context = num_context
        self.train = train
        self.action_dims = action_dims
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

        config_num = sample['infos'][0]['config_num']
        while len(batch['actions']) < batch_size:
            sample = self.reader.next()
            if sample['infos'][0]['config_num'] == config_num:
                for k, v in sample.items():
                    batch[k].append(v)
        return {k: np.concatenate(v)[:batch_size] for k, v in batch.items()}

    def on_epoch_end(self):
        self.num_context = np.random.randint(self.min_num_context, self.max_num_context)
        for _ in range(random.randint(0, 100)):
            self.reader.next()

    def __len__(self):
        return 1000000


class PolicyDataGenerator(keras.utils.Sequence):
    def __init__(self, savedir, batch_size, action_dims=1):
        """

        :param savedir:
        :param batch_size:
        """
        self.savedir = osp.join(DATA_DIR, savedir)
        self.reader = JsonReader(self.savedir)
        self.batch_size = batch_size
        self.action_dims = action_dims

    def __getitem__(self, index):
        batch = self.get_batch(self.batch_size)
        obs = batch['obs']
        actions = batch['actions'].reshape(-1, self.action_dims)
        return obs, actions

    def __len__(self):
        return 1000000

    def get_batch(self, batch_size):
        batch = defaultdict(list)
        sample = self.reader.next()
        for k, v in sample.items():
            batch[k].append(v)

        config_num = sample['infos'][0]['config_num']
        while len(batch['t']) < batch_size:
            sample = self.reader.next()
            if sample['infos'][0]['config_num'] == config_num:
                for k, v in sample.items():
                    batch[k].append(v)
        return {k: np.concatenate(v)[:batch_size] for k, v in batch.items()}


class CNPCartPoleData(keras.utils.Sequence):
    selected_keys = ['obs', 'new_obs', 'actions', 'infos']

    def __init__(self, savedir, batch_size, num_context, action_dims=1, train=False):
        self.savedir = osp.join(DATA_DIR, savedir)
        self.batch_size = batch_size
        self.min_num_context, self.max_num_context = num_context
        self.train = train
        self.action_dims = action_dims

        self.load_all()
        self.on_epoch_end()

    def load_all(self):
        self.data = {}
        data = defaultdict(list)
        files = glob.glob(self.savedir + '/*.json')
        for file in files:
            samples = self.load_file(file)
            for sample in samples:
                data[sample['infos']['config_num']].append(sample)
        for config_num, records in data.items():
            sub_data = defaultdict(list)
            for record in records:
                for k, v in record.items():
                    if k in ['obs', 'new_obs', 'actions']:
                        sub_data[k].append(v)
            for k, v in sub_data.items():
                sub_data[k] = np.concatenate(v, axis=0)

            self.data[config_num] = sub_data

    def shuffle(self):
        for _, v in self.data.items():
            idx = np.arange(len(v['actions']))
            np.random.shuffle(idx)
            for key, value in v.items():
                v[key] = value[idx]

    def load_file(self, filename):
        with open(filename, 'r') as fp:
            lines = fp.readlines()
            samples = [_from_json(ll) for ll in lines]
        samples = [self.slim_sample(sample) for sample in samples]
        return samples

    def slim_sample(self, sample: dict):
        slimed = {k: v for k, v in sample.items() if k in self.selected_keys}
        slimed['infos'] = slimed['infos'][0]
        return slimed

    def __getitem__(self, index):
        if self.train:
            batch = self.get_batch(index, max(self.batch_size, self.num_context))
        else:
            batch = self.get_batch(index, self.batch_size + self.num_context)

        obs = batch['obs']
        new_obs = batch['new_obs']
        actions = batch['actions'].reshape(new_obs.shape[0], -1)
        states = np.concatenate([obs, actions], axis=-1)
        delta = new_obs - obs

        if self.train:
            context_x = np.repeat(np.expand_dims(states[:self.num_context], axis=0),
                                  self.batch_size, axis=0)
            context_y = np.repeat(np.expand_dims(delta[:self.num_context], axis=0),
                                  self.batch_size, axis=0)
            query_x = states[-self.batch_size:]  # If train, include context points
            target_y = delta[-self.batch_size:]
        else:
            context_x = np.repeat(np.expand_dims(states[:self.num_context], axis=0),
                                  self.batch_size, axis=0)
            context_y = np.repeat(np.expand_dims(delta[:self.num_context], axis=0),
                                  self.batch_size, axis=0)
            query_x = states[-self.batch_size:]  # If test, select other points as query
            target_y = delta[-self.batch_size:]

        return {'context_x': context_x, 'context_y': context_y, 'query_x': query_x}, target_y

    def get_batch(self, i, batch_size):
        key = random.choice(list(self.data.keys()))
        batch = {k: v[i * batch_size: (i + 1) * batch_size] for k, v in self.data[key].items()}
        return batch

    def __len__(self):
        min_len = np.min([len(v['actions']) for v in self.data.values()])
        if self.train:
            return min_len // max(self.max_num_context, self.batch_size) - 1
        else:
            return min_len // (self.max_num_context + self.batch_size) - 1

    def on_epoch_end(self):
        self.num_context = np.random.randint(self.min_num_context, self.max_num_context)
        self.shuffle()


class PolicyData(keras.utils.Sequence):
    selected_keys = ['obs', 'actions']

    def __init__(self, savedir, batch_size):
        self.savedir = osp.join(DATA_DIR, savedir)
        self.batch_size = batch_size

        self.load_all()
        self.on_epoch_end()
        self.index = 0

    def load_all(self):
        data = defaultdict(list)
        files = glob.glob(self.savedir + '/*.json')
        for file in files:
            samples = self.load_file(file)
            for sample in samples:
                for k, v in sample.items():
                    data[k].append(v)
        for k, v in data.items():
            data[k] = np.concatenate(v, axis=0)
        self.data = data

    def shuffle(self):
        idx = np.arange(len(self.data['actions']))
        np.random.shuffle(idx)
        for k, v in self.data.items():
            self.data[k] = v[idx]

    def load_file(self, filename):
        with open(filename, 'r') as fp:
            lines = fp.readlines()
            samples = [_from_json(ll) for ll in lines]
        samples = [self.slim_sample(sample) for sample in samples]
        return samples

    def slim_sample(self, sample: dict):
        slimed = {k: v for k, v in sample.items() if k in self.selected_keys}
        return slimed

    def __getitem__(self, index):
        batch = {k: v[index * self.batch_size: (index + 1) * self.batch_size] for k, v in self.data.items()}
        return batch

    def __len__(self):
        return len(self.data['actions']) // self.batch_size - 1

    def on_epoch_end(self):
        self.shuffle()

    def next(self):
        batch = self.__getitem__(self.index)
        self.index += 1
        if self.index >= self.__len__():
            self.index = 0
        return batch


if __name__ == '__main__':
    cartpole_data_gen = CNPCartPoleData(savedir='offline/cartpole/random',
                                        batch_size=16,
                                        num_context=[10, 20],
                                        train=True)
    print(cartpole_data_gen.__getitem__(0)[0]['context_x'].shape)
    cartpole_data_gen = CNPCartPoleData(savedir='offline/cartpole/random',
                                        batch_size=16,
                                        num_context=[10, 20],
                                        train=False)
    print(cartpole_data_gen.__getitem__(0)[0]['context_x'].shape)

import random
import string
from collections import defaultdict

import numpy as np
import ray
from ray.rllib import rollout
from ray.rllib.agents import dqn
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from tqdm import tqdm
import matplotlib.pyplot as plt
from Algorithms.mpc import CemMPCController
from Data.basics import gen_context, remove_data, rollout_and_save_data
from Data.cartpole_data import CNPCartPoleGenerator
from Envs.custom_cartpole_v1 import CustomCartPole, make_cartpole_reward_on_traj
from Models.cnp_model import CNPModel
from Models.mlp_model import MLPModel
from Models.policy_model import PolicyFCModel
from sklearn.preprocessing import OneHotEncoder

NUM_CONFIGS = 50
NEW_DATA = False
LEARN_DYNAMICS = False

if __name__ == '__main__':
    dummy_env = CustomCartPole({'masscart': 1.0,
                                'masspole': 0.1,
                                'length': 0.5,
                                'force_mag': 10, })

    # 1. Collect Data for Dynamics Learning (Random Policy, Random Physics Config)
    if NEW_DATA:
        print('------------------------------')
        print('Collect Random Data for Dynamics Learning')
        lengths = np.round(np.random.uniform(low=0.5, high=1.5, size=NUM_CONFIGS), decimals=2)
        env_configs = [{'masscart': 1.0,
                        'masspole': 0.1,
                        'length': l,
                        'force_mag': 10, } for l in lengths]

        remove_data('offline/cartpole/random')  # clean existing data
        rollout_and_save_data(CustomCartPole,
                              savedir='offline/cartpole/random',
                              env_configs=env_configs,
                              episodes=500,
                              max_steps_per_episode=200,
                              controller=None)

    # 2. Learn Dynamics CNP
    print('------------------------------')
    print('Learn Dynamics CNP ')

    # action_enc = OneHotEncoder()
    # action_enc.fit(np.arange(dummy_env.action_space.n).reshape(-1, 1))

    cartpole_dynamics = CNPModel(
        state_dims=4,
        action_dims=1,
        name='cartpole_dynamics',
    )
    if LEARN_DYNAMICS:
        data_generator = CNPCartPoleGenerator(savedir='offline/cartpole/random',
                                              batch_size=16,
                                              num_context=[15, 25],
                                              train=True)
        cartpole_dynamics.build_model(encoder_hiddens=[128, 128, 128, 128], decoder_hiddens=[128, 128])
        cartpole_dynamics.train(data_generator, epochs=100)
    else:
        cartpole_dynamics.load_model()

    # Evaluate Dynamics with different configs
    configs = [{'masscart': 1.0,
                'masspole': 0.1,
                'length': l,
                'force_mag': 10, } for l in np.arange(0.5, 1.5, 0.1)]

    # Unify context points
    context_actions = [dummy_env.action_space.sample() for _ in range(20)]
    rollout_actions = [dummy_env.action_space.sample() for _ in range(100)]

    results = {}

    for config in configs:
        env = CustomCartPole(config)
        context_x, context_y = gen_context(env, num_context_points=20, actions=context_actions)
        cartpole_dynamics.set_context(context_x, context_y)

        # train MLP model for comparation
        mlp_model = MLPModel(state_dims=4, action_dims=1)
        mlp_model.build_model([128, 128, 128, 128])
        mlp_model.train(context_x, context_y)

        results[config['length']] = defaultdict(list)

        # Rollout
        obs = env.reset()
        state = obs.copy()
        mlp_state = obs.copy()

        for i in tqdm(range(len(rollout_actions))):
            # gt
            new_obs, rew, done, info = env.step(rollout_actions[i])
            results[config['length']]['gt'].append(new_obs.copy())

            query_x = np.concatenate([state, np.array([rollout_actions[i]])]).reshape(1, -1)
            target_y = cartpole_dynamics.predict(query_x)
            state += target_y['mu'].flatten()
            sigma = target_y['sigma'].flatten()
            results[config['length']]['mu'].append(state.copy())
            results[config['length']]['sigma'].append(sigma.copy())

            query_x = np.concatenate([mlp_state, np.array([rollout_actions[i]])]).reshape(1, -1)
            target_y = mlp_model.predict(query_x)
            mlp_state += target_y.flatten()
            results[config['length']]['mlp'].append(mlp_state.copy())

            if done:
                break

        # display
        fig = plt.figure()
        fig.suptitle(f"Length: {config['length']}")
        for i in range(dummy_env.observation_space.shape[0]):
            plt.subplot(221 + i)
            axes = plt.gca()
            axes.set_ylim(-1, 1)
            xs = np.arange(len(results[config['length']]['gt']))
            plt.plot(xs, np.stack(results[config['length']]['gt'])[:, i], 'r')
            plt.plot(xs, np.stack(results[config['length']]['mlp'])[:, i], 'g')
            plt.plot(xs, np.stack(results[config['length']]['mu'])[:, i], 'b')
            plt.fill_between(xs,
                             np.stack(results[config['length']]['mu'])[:, i] + np.stack(results[config[
                                 'length']]['sigma'])[:, i],
                             np.stack(results[config['length']]['mu'])[:, i] - np.stack(results[config[
                                 'length']]['sigma'])[:, i],
                             alpha=0.2,
                             facecolor='#65c9f7',
                             )
            print(np.stack(results[config[
                'length']]['sigma'])[:, i])

        plt.show()

    # # display
    # for i in range(dummy_env.observation_space.shape[0]):
    #     fig = plt.figure()
    #     fig.suptitle(f"Channel {i}")
    #     for k, v in results.items():
    #         plt.plot(np.stack(v['gt'])[:, i], label='gt', color='g')
    #         plt.plot(np.stack(v['cnp'])[:, i], label='cnp', color='b')
    #     plt.show()

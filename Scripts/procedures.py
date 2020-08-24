import random
import string

import numpy as np
import ray
from ray.rllib import rollout
from ray.rllib.agents import dqn
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from tqdm import tqdm

from Algorithms.mpc import CemMPCController
from Data import DATA_DIR
from Data.basics import gen_context, remove_data, rollout_and_save_data
from Data.cartpole_data import CNPCartPoleGenerator
from Envs.custom_cartpole_v1 import CustomCartPole, make_cartpole_reward_on_traj
from Models.cnp_model import CNPModel
from Models.policy_model import PolicyFCModel
import os.path as osp
import json


def collect_data_for_dynamics(num_dynamics):
    print('------------------------------')
    print('Collect Random Data for Dynamics Learning')
    lengths = np.round(np.random.uniform(low=0.5, high=1.0, size=num_dynamics), decimals=2)
    env_configs = [{'masscart': 1.0,
                    'masspole': 0.1,
                    'length': l,
                    'force_mag': 10, } for l in lengths]

    remove_data('offline/cartpole/random')  # clean existing data
    rollout_and_save_data(CustomCartPole,
                          savedir='offline/cartpole/random',
                          env_configs=env_configs,
                          episodes=100,
                          max_steps_per_episode=200,
                          controller=None)


def create_cnp_dynamics(train=False, savedir='offline/cartpole/random'):
    print('------------------------------')
    print('Learn Dynamics CNP ')
    cartpole_dynamics = CNPModel(
        state_dims=4,
        action_dims=1,
        name='cartpole_dynamics',
    )
    if train:
        print(f'Train dynamics from data: {savedir}')
        data_generator = CNPCartPoleGenerator(savedir=savedir,
                                              batch_size=16,
                                              num_context=[10, 20],
                                              train=True)
        cartpole_dynamics.build_model(encoder_hiddens=[32, 16], decoder_hiddens=[16, 32])
        cartpole_dynamics.train(data_generator, epochs=100)
    else:
        print('Load dynamics without training')
        cartpole_dynamics.load_model()

    return cartpole_dynamics


def randomly_choose_an_environment(use_old=False):
    print('------------------------------')
    if use_old:
        with open(osp.join(DATA_DIR, 'intermediate/env_config.txt'), 'r') as fp:
            chosen_config = json.load(fp)
            print(f'Load old config of length: {chosen_config["length"]}')

    else:
        chosen_config = {'masscart': 1.0,
                         'masspole': 0.1,
                         'length': np.round(np.random.uniform(low=0.5, high=1.0), decimals=2),
                         'force_mag': 10, }
        print(f'Choose new environment of length: {chosen_config["length"]}')

    chosen_env = CustomCartPole(chosen_config)
    reward_fn = make_cartpole_reward_on_traj(chosen_env)
    with open(osp.join(DATA_DIR, 'intermediate/env_config.txt'), 'w') as fp:
        json.dump(chosen_config, fp)

    return chosen_env, reward_fn


def sample_context_points_from_env(env, num_context_points=15):
    print('------------------------------')
    print(f'Collect {num_context_points} Context Points')
    context_x, context_y = gen_context(env, num_context_points=num_context_points)
    return context_x, context_y


def create_MPC_controller(controller_cls, dynamics, env, reward_fn, horizon=20, num_samples=20):
    print('------------------------------')
    controller = controller_cls(action_space=env.action_space,
                                action_dims=1,
                                dynamics=dynamics,
                                reward_fn=reward_fn,
                                horizon=horizon,
                                samples=num_samples)
    print(f'Created controller {controller.name}')

    return controller


def test_controller_by_rolling_out(controller, env, render=False):
    print('------------------------------')
    print("Rollout Once to Test Controller")

    if isinstance(env, dict):
        env = CustomCartPole(env)

    accumulated_reward = 0
    obs = env.reset()
    for _ in tqdm(range(200)):
        action = int(controller.next_action(obs))
        obs, rew, done, info = env.step(int(action))
        accumulated_reward += rew
        if render:
            env.render()
        if done:
            break
    print(f'Reward: {accumulated_reward}')


def collect_mpc_data(controller, env, num_episodes=20):
    print('------------------------------')
    print(f"MPC Controller Rollout to Collect Supervised Data of {num_episodes} Episodes")
    remove_data(f'offline/cartpole/{controller.name}')  # clean existing data
    rollout_and_save_data(CustomCartPole,
                          savedir=f'offline/cartpole/{controller.name}',
                          env_configs=[env.config],
                          episodes=num_episodes,
                          max_steps_per_episode=200,
                          controller=controller)


def create_rllib_trainer(env, controller, env_name, model_name, trainer_cls=dqn.DQNTrainer):
    print('----------------------------------------------------')
    print(f'Model name: {model_name}, Env name: {env_name}')
    register_env(env_name, lambda config: CustomCartPole(config))

    ModelCatalog.register_custom_model(model_name, PolicyFCModel)

    model_config = {
        'custom_model': model_name,
        "custom_model_config": {
            'hiddens': [256, 128, 16],
            'offline_dataset': f'offline/cartpole/{controller.name}',
        }
    }
    rl_config = {
        "train_batch_size": 200,
        'num_workers': 0,
        'env_config': env.config,
        'model': model_config,
    }
    trainer = trainer_cls(config=rl_config, env='MPC-BC-v1')
    print(f'Created trainer')  # TODO: name
    return trainer


def supervised_learning_for_initialization(trainer, train=True, batch_size=16):
    print('----------------------------------------------------')
    print('Supervised learning for good initialization')
    trainer.get_policy().model.initialization(train, batch_size=batch_size)


def supervised_loss_for_behavior_cloning(trainer, train_steps=10):
    print('------------------------------')
    results = []
    for i in tqdm(range(train_steps)):
        result_mf = trainer.train()
        print(f"\t RL Reward: "
              f"{result_mf['episode_reward_max']:.4f}  |  "
              f"{result_mf['episode_reward_mean']:.4f}  |  "
              f"{result_mf['episode_reward_min']:.4f}")
        results.append(result_mf['episode_reward_mean'])
        if i % 5 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
        if i % 5 == 0:
            rollout.rollout(trainer, env_name='CustomCartPole-v1', num_steps=50, num_episodes=1, no_render=False)
    return results

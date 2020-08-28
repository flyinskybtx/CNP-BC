import json
import os.path as osp
from collections import defaultdict
import matplotlib.pyplot as plt

import numpy as np
from ray.rllib import rollout
from ray.rllib.agents import dqn
from ray.rllib.evaluation import SampleBatchBuilder
from ray.rllib.models import ModelCatalog
from ray.rllib.offline import JsonWriter
from ray.rllib.rollout import RolloutSaver
from ray.rllib.utils import try_import_tf
from ray.tune import register_env

from Models.mlp_model import MLPModel

_, tf, _ = try_import_tf()
from tqdm import tqdm

from Data import DATA_DIR
from Data.basics import gen_context, remove_data, rollout_and_save_data
from Data.cartpole_data import CNPCartPoleGenerator, CNPCartPoleData
from Envs.custom_cartpole_v1 import CustomCartPole, make_cartpole_reward_on_traj
from Models import MODEL_DIR
from Models.cnp_model import CNPModel
from Models.policy_model import build_fc_model, logits_dist_loss, logits_cate_acc, SurrogateFCModel


def collect_data_for_dynamics(num_dynamics, episodes=100):
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
                          episodes=episodes,
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
        train_generator = CNPCartPoleData(savedir='offline/cartpole/random',
                                          batch_size=16,
                                          num_context=[10, 20],
                                          train=True)
        vali_generator = CNPCartPoleData(savedir='offline/cartpole/random',
                                         batch_size=16,
                                         num_context=[10, 20],
                                         train=True)
        cartpole_dynamics.build_model(encoder_hiddens=[32, 16], decoder_hiddens=[16, 32])
        cartpole_dynamics.train_model(train_generator, vali_generator, epochs=100)
        cartpole_dynamics.save_model()
    else:
        print('Load dynamics without training')
        cartpole_dynamics.load_model()

    return cartpole_dynamics


def randomly_choose_an_environment(use_new=False):
    print('------------------------------')
    if not use_new:
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


def create_rllib_trainer(env, controller, model_cls, env_name, model_name, trainer_cls=dqn.DQNTrainer):
    print('----------------------------------------------------')
    print(f'Model name: {model_name}, Env name: {env_name}')
    register_env(env_name, lambda config: CustomCartPole(config))

    ModelCatalog.register_custom_model(model_name, model_cls)

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
    agent = trainer_cls(config=rl_config, env='MPC-BC-v1')
    print(f'Created trainer')  # TODO: name
    return agent


def create_surrogate_model(agent):
    surrogate_model = SurrogateFCModel(agent.get_policy().model.obs_space,
                                       agent.get_policy().model.action_space,
                                       agent.get_policy().model.hiddens,
                                       agent.get_policy().model.model_config['custom_model_config']['offline_dataset'],
                                       name=agent.get_policy().model.base_model.name)
    return surrogate_model


def behavior_cloning(agent, train_steps=2, weights_file=None):
    print('------------------------------')
    agent.get_policy().model.refresh_reader()  # Update reader
    results = []
    for i in range(train_steps):
        result_mf = agent.train()
        print(f"RL Reward: \t"
              f"{result_mf['episode_reward_max']:.4f}  |  "
              f"{result_mf['episode_reward_mean']:.4f}  |  "
              f"{result_mf['episode_reward_min']:.4f}")
        results.append(result_mf['episode_reward_mean'])
        if i % 5 == 0:
            # checkpoint = agent.save()
            # print("\ncheckpoint saved at", checkpoint)
            with agent.get_policy()._sess.graph.as_default():
                with agent.get_policy()._sess.as_default():
                    agent.get_policy().model.base_model.save_weights(weights_file)
        # if i % 5 == 0:
        #     rollout.rollout(agent, env_name='CustomCartPole-v1', num_steps=50, num_episodes=1, no_render=False)
    return results


def supervised_initialization(agent, surrogate, epochs=100):
    print('----------------------------------------------------')
    print('Supervised learning for good initialization')
    surrogate.load_weights()
    surrogate.train(epochs)
    surrogate.save_weights()
    agent.get_policy().import_model_from_h5(surrogate.weights_file)


def policy_rollout(agent, env_name, savedir=f'offline/cartpole/rollout', episodes=10, ):
    print('----------------------------------------------------')
    print('Use policy to rollout for dynamics fine-tune')
    savedir = osp.join(DATA_DIR, savedir)
    saver = RolloutSaver(osp.join(DATA_DIR, 'rollout'))
    batch_builder = SampleBatchBuilder()
    writer = JsonWriter(f"{savedir}")

    for _ in tqdm(range(episodes)):
        rollout.rollout(agent, env_name=env_name,
                        num_steps=50, num_episodes=1, no_render=True,
                        saver=saver)
        saver.end_rollout()
        for data in saver._rollouts[0]:
            obs, action, new_obs, reward, done = data
            batch_builder.add_values(
                obs=obs,
                new_obs=new_obs,
                actions=action,
                dones=done,
                reward=reward,
                infos={'config_num': 0}
            )
        writer.write(batch_builder.build_and_reset())


def examine_dynamics(dynamics, tracks, mlp_comparison=False):
    # train MLP model for comparation
    if mlp_comparison:
        mlp_model = MLPModel(state_dims=4, action_dims=1)
        mlp_model.build_model([128, 128, 128, 128])
        mlp_model.train(dynamics.context_x, dynamics.context_y)

    results = defaultdict(list)
    for track in tracks:
        state = track['obs'].copy()
        results['gt'] += track['gt']
        if mlp_comparison:
            mlp_state = state.copy()
        for action in track['actions']:
            query_x = np.concatenate([state, np.array([action])]).reshape(-1, 1)
            target_y = dynamics.predict(query_x)
            state += target_y['mu'].flatten()
            sigma = target_y['sigma'].flatten()
            results['mu'].append(state.copy())
            results['sigma'].append(sigma.copy())

            if mlp_comparison:
                query_x = np.concatenate([mlp_state, np.array([action])]).reshape(-1, 1)
                target_y = mlp_model.predict(query_x)
                mlp_state += target_y.flatten()
                results['mlp'].append(mlp_state.copy())

    # Estimate Error
    mu = np.stack(results['mu'], axis=0)
    gt = np.stack(results['gt'], axis=0)
    mse = np.mean(np.sum(np.square(mu - gt), axis=-1))
    print(f'MSE Error: {mse}')

    # Display
    fig = plt.figure()
    fig.suptitle(f"Dynamics evaluation, MSE: {mse}")
    for i in range(4):
        plt.subplot(221 + i)
        axes = plt.gca()
        axes.set_ylim(-1, 1)
        xs = np.arange(len(results['gt']))
        l1 = plt.plot(xs, np.stack(results['gt'])[:, i], 'r')
        l2 = plt.plot(xs, np.stack(results['mu'])[:, i], 'b')
        plt.fill_between(xs,
                         np.stack(results['mu'])[:, i] + np.stack(results['sigma'])[:, i],
                         np.stack(results['mu'])[:, i] - np.stack(results['sigma'])[:, i],
                         alpha=0.2,
                         facecolor='#65c9f7',
                         )
        if mlp_comparison:
            plt.plot(xs, np.stack(results['mlp'])[:, i], 'g')
        plt.legend()
    plt.show()


def sample_examine_track(env, episodes, horizon=20):
    tracks = []
    for i in range(episodes):
        step = 0
        obs = env.reset()
        cur_track = {'obs': obs, 'actions': [], 'gt': []}
        done = False
        while not done and step < horizon:
            action = env.action_space.sample()
            new_obs, rew, done, info = env.step(action)
            cur_track['actions'].append(action)
            cur_track['gt'].append(new_obs)
            step += 1
        tracks.append(cur_track)
    return tracks


def examine_dynamics_with_configuration_changes(dynamics, env_cls, actions, num_context):
    # Evaluate Dynamics with different configs
    configs = [{'masscart': 1.0,
                'masspole': 0.1,
                'length': l,
                'force_mag': 10, } for l in np.arange(0.5, 1.0, 0.05)]

    results = {}

    for config in configs:
        env = env_cls(config)
        context_x, context_y = gen_context(env, num_context_points=num_context, actions=actions)
        dynamics.set_context(context_x, context_y)

        # train MLP model for comparation
        mlp_model = MLPModel(state_dims=4, action_dims=1)
        mlp_model.build_model([128, 128, 128, 128])
        mlp_model.train(context_x, context_y)

        results[config['length']] = defaultdict(list)

        # Rollout
        obs = env.reset()
        state = obs.copy()
        mlp_state = obs.copy()

        for i in tqdm(range(len(actions))):
            # gt
            new_obs, rew, done, info = env.step(actions[i])
            results[config['length']]['gt'].append(new_obs.copy())

            query_x = np.concatenate([state, np.array([actions[i]])]).reshape(1, -1)
            target_y = dynamics.predict(query_x)
            state += target_y['mu'].flatten()
            sigma = target_y['sigma'].flatten()
            results[config['length']]['mu'].append(state.copy())
            results[config['length']]['sigma'].append(sigma.copy())

            query_x = np.concatenate([mlp_state, np.array([actions[i]])]).reshape(1, -1)
            target_y = mlp_model.predict(query_x)
            mlp_state += target_y.flatten()
            results[config['length']]['mlp'].append(mlp_state.copy())

            if done:
                break

        # display
        fig = plt.figure()
        fig.suptitle(f"Length: {config['length']}")
        for i in range(4):
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

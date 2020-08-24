import random
import string

import matplotlib.pyplot as plt
import numpy as np
import ray
from ray.rllib import rollout
from ray.rllib.agents import a3c
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from tqdm import tqdm

from Algorithms.mpc import NaiveMPCController
from Data.cartpole_data import CNPCartPoleGenerator
from Envs.custom_cartpole_v1 import CustomCartPole, make_cartpole_reward
from Models.cnp_model import CNPModel
from Models.policy_model import PolicyFCModel

if __name__ == '__main__':
    train_dynamics = False
    gen_new_data = True
    dynamics_rollout = False
    mpc_rollout = False

    env_config = {
        'masscart': 1.0,
        'masspole': 0.1,
        'length': np.random.uniform(0.5, 1),
        'force_mag': 10,
    }

    cartpole_env = CustomCartPole(env_config)

    cartpole_dynamics = CNPModel(
        state_dims=cartpole_env.observation_space.shape[0],
        action_dims=1,
        name='cartpole_dynamics',
    )

    # ------------------- DYNAMICS -------------------- #
    if train_dynamics:
        cartpole_data_gen = CNPCartPoleGenerator(savedir='offline/cartpole', batch_size=16, num_context=[10, 20])
        if gen_new_data:
            cartpole_data_gen.random_rollout_and_save_data(num_configs=1, episodes=10000, steps_per_episode=1000)
        cartpole_dynamics.build_model(encoder_hiddens=[32, 16], decoder_hiddens=[16, 32])
        cartpole_dynamics.train(cartpole_data_gen, epochs=100)
    else:
        cartpole_dynamics.load_model()

    # ------------------- Generate Context ------------------------- #
    results = {'gt': [], 'cnp': [], 'context': []}

    # Get context points
    context_x = []
    context_y = []
    num_context_points = 15

    while len(context_x) < num_context_points:
        obs = cartpole_env.reset()
        for i in range(100):
            action = cartpole_env.action_space.sample()
            new_obs, rew, done, info = cartpole_env.step(action)
            delta = new_obs - obs
            context_x.append(np.concatenate([obs, np.array([action])]))
            context_y.append(delta)
            obs = new_obs
            results['context'].append(new_obs)
            if done:
                break

    idx = np.arange(len(context_x))
    np.random.shuffle(idx)
    context_x = np.stack(context_x, axis=0)[idx[:num_context_points]]
    context_y = np.stack(context_y, axis=0)[idx[:num_context_points]]
    cartpole_dynamics.set_context(context_x, context_y)

    # ------------------- Dynamics Rollout ------------------------- #
    # rollout
    if dynamics_rollout:
        obs = cartpole_env.reset()
        state = obs.copy()
        for i in tqdm(range(100)):
            # gt
            action = cartpole_env.action_space.sample()
            new_obs, rew, done, info = cartpole_env.step(action)
            results['gt'].append(new_obs.copy())

            # cnp
            query_x = np.concatenate([state, np.array([action])]).reshape(1, -1)
            target_y = cartpole_dynamics.predict(query_x)
            state += target_y['mu'].flatten()
            results['cnp'].append(state.copy())

            if done:
                break

        # display
        fig = plt.figure()
        fig.set_label('Dynamics Error')
        for i in range(cartpole_env.observation_space.shape[0]):
            plt.subplot(221 + i)
            axes = plt.gca()
            axes.set_ylim(-1, 1)
            plt.plot(np.stack(results['gt'])[:, i], 'b')
            plt.plot(np.stack(results['cnp'])[:, i], 'g')

        plt.show()

    # ------------------- MPC Rollout ------------------------- #
    controller = NaiveMPCController(action_space=cartpole_env.action_space,
                                    dynamics=cartpole_dynamics,
                                    reward_fn=make_cartpole_reward(cartpole_env),
                                    horizon=100,
                                    samples=10, )
    if mpc_rollout:
        accumulated_reward = 0
        obs = cartpole_env.reset()
        for i in tqdm(range(200)):
            action = int(controller.next_action(obs)[0])
            obs, rew, done, info = cartpole_env.step(action)
            accumulated_reward += rew
            print(accumulated_reward)
            cartpole_env.render()
            if done:
                break

        # evaluate dynamics error
        results = {'gt': [], 'dynamics': []}
        obs = cartpole_env.reset()
        state = obs.copy()
        actions = controller.next_action(obs)
        actions = [int(aa) for aa in actions]
        for i, action in enumerate(actions):
            obs, rew, done, info = cartpole_env.step(action)
            results['gt'].append(obs)
            t_f = i
            if done:
                break
        for i, action in enumerate(actions):

            query_x = np.concatenate([state, np.array([action])]).reshape(1, -1)
            target_y = cartpole_dynamics.predict(query_x)
            state += target_y['mu'].flatten()
            results['dynamics'].append(state.copy())
            if i >= t_f:
                break

        # display
        fig = plt.figure()
        fig.set_label('MPC error')
        for i in range(cartpole_env.observation_space.shape[0]):
            plt.subplot(221 + i)
            axes = plt.gca()
            axes.set_ylim(-1, 1)
            plt.plot(np.stack(results['gt'])[:, i], 'b')
            plt.plot(np.stack(results['dynamics'])[:, i], 'g')
        plt.show()


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

from Algorithms.mpc import NaiveMPCController, CemMPCController
from Data.basics import gen_context, rollout_and_save_data
from Data.cartpole_data import CNPCartPoleGenerator
from Envs.custom_cartpole_v1 import CustomCartPole, make_cartpole_reward_on_traj
from Models.cnp_model import CNPModel
from Models.policy_model import PolicyFCModel

if __name__ == '__main__':
    env_config = {
        'masscart': 1.0,
        'masspole': 0.1,
        'length': np.random.uniform(0.5, 1),
        'force_mag': 10,
    }

    cartpole_env = CustomCartPole(env_config)
    reward_fn = make_cartpole_reward_on_traj(cartpole_env)

    cartpole_dynamics = CNPModel(
        state_dims=cartpole_env.observation_space.shape[0],
        action_dims=1,
        name='cartpole_dynamics',
    )

    cartpole_dynamics.load_model()

    num_context_points = 15
    context_x, context_y = gen_context(cartpole_env, num_context_points)
    cartpole_dynamics.set_context(context_x, context_y)

    controller = CemMPCController(action_space=cartpole_env.action_space,
                                  action_dims=1,
                                  dynamics=cartpole_dynamics,
                                  reward_fn=reward_fn,
                                  horizon=20,
                                  samples=20)

    # ------------- start rollout ------------------- #
    accumulated_reward = 0
    obs = cartpole_env.reset()
    for i in tqdm(range(200)):
        action = controller.next_action(obs, print_expectation=True)
        obs, rew, done, info = cartpole_env.step(int(action))
        accumulated_reward += rew
        print(f'Current_reward: {accumulated_reward}, State: {obs}')
        cartpole_env.render()
        if done:
            break

    # rollout_and_save_data(num_configs=1,
    #                       episodes=100,
    #                       steps_per_episode=1000,
    #                       controller=controller,
    #                       savedir=f'offline/cartpole/{controller.name}')

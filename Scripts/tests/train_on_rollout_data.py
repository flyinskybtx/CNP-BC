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
    train_data = CNPCartPoleGenerator(savedir='offline/cartpole/rollout', batch_size=16, num_context=[10, 20])
    vali_data = CNPCartPoleGenerator(savedir='offline/cartpole/rollout', batch_size=16, num_context=[10, 20])

    cartpole_dynamics.build_model(encoder_hiddens=[32, 16], decoder_hiddens=[16, 32])
    cartpole_dynamics.train_model(train_data, vali_data, epochs=100)

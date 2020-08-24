import random
import string

import numpy as np
import ray
from ray.rllib import rollout
from ray.rllib.agents import dqn
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from tqdm import tqdm

from Algorithms.mpc import CemMPCController, NaiveMPCController
from Data.basics import gen_context, remove_data, rollout_and_save_data
from Data.cartpole_data import CNPCartPoleGenerator
from Envs.custom_cartpole_v1 import CustomCartPole, make_cartpole_reward_on_traj
from Models.cnp_model import CNPModel
from Models.policy_model import PolicyFCModel
from Scripts.procedures import collect_data_for_dynamics, create_cnp_dynamics, randomly_choose_an_environment, \
    sample_context_points_from_env, create_MPC_controller, test_controller_by_rolling_out, collect_mpc_data, \
    create_rllib_trainer, supervised_learning_for_initialization, supervised_loss_for_behavior_cloning

COLLECT_NEW_DATA = False
USE_OLD_CONFIG = False
NUM_DYNAMICS = 20
TRAIN_DYNAMICS = False
NUM_CONTEXT_POINTS = 15
HORIZON = 20
NUM_MPC_SAMPLES = 50
TEST_MPC = False
MPC_EPISODES = 20
MPC_CLS = NaiveMPCController  # CemMPCController
MODEL_NAME = 'CartPoleDQNModel'
ENV_NAME = 'MPC-BC-v1'

if __name__ == '__main__':
    ray.shutdown(True)
    ray.init(num_gpus=1)

    if COLLECT_NEW_DATA:
        collect_data_for_dynamics(NUM_DYNAMICS)

    cartpole_dynamics = create_cnp_dynamics(TRAIN_DYNAMICS, savedir='offline/cartpole/random')

    chosen_env, reward_fn = randomly_choose_an_environment(USE_OLD_CONFIG)

    context_x, context_y = sample_context_points_from_env(chosen_env, NUM_CONTEXT_POINTS)
    cartpole_dynamics.set_context(context_x, context_y)

    mpc_controller = create_MPC_controller(MPC_CLS, cartpole_dynamics, chosen_env,
                                           reward_fn, HORIZON, NUM_MPC_SAMPLES)

    if TEST_MPC:
        test_controller_by_rolling_out(mpc_controller, chosen_env, render=True)

    # ----------
    trainer = create_rllib_trainer(chosen_env, mpc_controller, env_name=ENV_NAME, model_name=MODEL_NAME,
                                   trainer_cls=dqn.DQNTrainer)
    for i in range(10):
        collect_mpc_data(mpc_controller, chosen_env, num_episodes=MPC_EPISODES)
        supervised_learning_for_initialization(trainer, train=True, batch_size=16)
        results = supervised_loss_for_behavior_cloning(trainer)
        print(results)


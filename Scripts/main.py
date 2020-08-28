import random
import string
import time

import ray
from ray.rllib.agents import a3c, pg

from Algorithms.mpc import NaiveMPCController, CemMPCController
from Data.basics import remove_data
from Data.cartpole_data import CNPCartPoleGenerator, CNPCartPoleData
from Envs.custom_cartpole_v1 import CustomCartPole
from Models.policy_model import PolicyFCModel
from Scripts.procedures import collect_data_for_dynamics, create_cnp_dynamics, randomly_choose_an_environment, \
    sample_context_points_from_env, create_MPC_controller, test_controller_by_rolling_out, create_rllib_trainer, \
    behavior_cloning, create_surrogate_model, \
    supervised_initialization, policy_rollout, collect_mpc_data, examine_dynamics, sample_examine_track, \
    examine_dynamics_with_configuration_changes

COLLECT_NEW_DATA = True
USE_NEW_CONFIG = True
TRAIN_DYNAMICS = True
TEST_MPC = True

NUM_DYNAMICS = 20
NUM_RANDOM_EPISODES = 500
NUM_CONTEXT_POINTS = 15
HORIZON = 50
NUM_MPC_SAMPLES = 20
MPC_EPISODES = 5
SUPERVISED_EPOCHS = 100
BEHAVIOR_CLONE_EPISODES = 20
MPC_CLS = CemMPCController  # NaiveMPCController
DYNAMICS_EXAMINE_STEPS = 100

MODEL_NAME = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
ENV_NAME = 'MPC-BC-v1'

if __name__ == '__main__':
    ray.shutdown(True)
    ray.init(num_gpus=1)

    if COLLECT_NEW_DATA:
        collect_data_for_dynamics(NUM_DYNAMICS, episodes=NUM_RANDOM_EPISODES)

    cartpole_dynamics = create_cnp_dynamics(TRAIN_DYNAMICS, savedir='offline/cartpole/random')

    chosen_env, reward_fn = randomly_choose_an_environment(USE_NEW_CONFIG)
    examine_tracks = sample_examine_track(chosen_env, episodes=10, horizon=HORIZON)

    examine_dynamics_with_configuration_changes(cartpole_dynamics, CustomCartPole, examine_tracks[0]['actions'],
                                                num_context=NUM_CONTEXT_POINTS)

    context_x, context_y = sample_context_points_from_env(chosen_env, NUM_CONTEXT_POINTS)
    cartpole_dynamics.set_context(context_x, context_y)

    examine_dynamics(cartpole_dynamics, examine_tracks, mlp_comparison=False)

    mpc_controller = create_MPC_controller(MPC_CLS, cartpole_dynamics, chosen_env,
                                           reward_fn, HORIZON, NUM_MPC_SAMPLES)

    # ----------
    agent = create_rllib_trainer(chosen_env, mpc_controller, model_cls=PolicyFCModel, env_name=ENV_NAME,
                                 model_name=MODEL_NAME,
                                 # trainer_cls=dqn.DQNTrainer
                                 # trainer_cls=a3c.A2CTrainer
                                 trainer_cls=pg.PGTrainer
                                 )
    surrogate_model = create_surrogate_model(agent)

    # TODO: Clean up existing data on old config
    remove_data('offline/cartpole/rollout')

    for i in range(10):
        if TEST_MPC:
            print('---------------------------------------')
            print('Test MPC')
            test_controller_by_rolling_out(mpc_controller, chosen_env, render=True)

        # Collect data from MPC
        print('---------------------------------------')
        print('Collect data from MPC')
        remove_data(f'offline/cartpole/{mpc_controller.name}')
        collect_mpc_data(mpc_controller, chosen_env, num_episodes=MPC_EPISODES)

        # Behavior clone
        print('---------------------------------------')
        print('A2C learning')
        supervised_initialization(agent, surrogate_model, epochs=SUPERVISED_EPOCHS)
        results = behavior_cloning(agent, train_steps=BEHAVIOR_CLONE_EPISODES,
                                   weights_file=surrogate_model.weights_file)
        print(results)

        # Use agent to generate new data
        print('---------------------------------------')
        print('Use agent to generate new data')
        policy_rollout(agent, env_name=ENV_NAME, savedir='offline/cartpole/rollout', episodes=MPC_EPISODES)
        train_data = CNPCartPoleData(savedir='offline/cartpole/rollout', batch_size=16, num_context=[10, 20],
                                     train=True)
        vali_data = CNPCartPoleData(savedir='offline/cartpole/rollout', batch_size=16, num_context=[10, 20],
                                    train=True)

        # Re-train dynamics
        print('---------------------------------------')
        print('Re-train dynamics')
        cartpole_dynamics.train_model(train_data, vali_data, epochs=20)
        examine_dynamics(cartpole_dynamics, examine_tracks, mlp_comparison=False)

        # update MPC
        print('---------------------------------------')
        print('Update MPC')
        mpc_controller.dynamics = cartpole_dynamics

        # TODO: evaluate MPC

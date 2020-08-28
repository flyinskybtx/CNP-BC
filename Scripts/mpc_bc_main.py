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
from Data.basics import gen_context, remove_data, rollout_and_save_data
from Data.cartpole_data import CNPCartPoleGenerator
from Envs.custom_cartpole_v1 import CustomCartPole, make_cartpole_reward_on_traj
from Models.cnp_model import CNPModel
from Models.policy_model import PolicyFCModel

NUM_CONFIGS = 50
LEARN_DYNAMICS = False

if __name__ == '__main__':
    ray.shutdown(True)
    ray.init(num_gpus=1)

    # 1. Collect Data for Dynamics Learning (Random Policy, Random Physics Config)
    print('------------------------------')
    print('Collect Random Data for Dynamics Learning')
    lengths = np.random.uniform(low=0.5, high=1.0, size=NUM_CONFIGS)
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

    # 2. Learn Dynamics CNP
    print('------------------------------')
    print('Learn Dynamics CNP ')
    cartpole_dynamics = CNPModel(
        state_dims=4,
        action_dims=1,
        name='cartpole_dynamics',
    )
    if LEARN_DYNAMICS:
        train_generator = CNPCartPoleGenerator(savedir='offline/cartpole/random',
                                              batch_size=16,
                                              num_context=[10, 20],
                                              train=True)
        vali_generator = CNPCartPoleGenerator(savedir='offline/cartpole/random',
                                              batch_size=16,
                                              num_context=[10, 20],
                                              train=True)
        cartpole_dynamics.build_model(encoder_hiddens=[32, 16], decoder_hiddens=[16, 32])
        cartpole_dynamics.train_model(train_generator, vali_generator, epochs=100)
    else:
        cartpole_dynamics.load_model()

    # 3.0 Pick a Random Physics
    print('------------------------------')
    chosen_config = {'masscart': 1.0,
                     'masspole': 0.1,
                     'length': np.random.uniform(low=0.5, high=1.0),
                     'force_mag': 10, }
    chosen_env = CustomCartPole(chosen_config)
    reward_fn = make_cartpole_reward_on_traj(chosen_env)

    # 3.1 Collect Context for Dynamics Prediction (Random Policy, Fixed Physics)
    print('Collect Context Points')
    context_x, context_y = gen_context(chosen_env, num_context_points=15)
    cartpole_dynamics.set_context(context_x, context_y)

    # 4.0 Create MPC Controller
    print('------------------------------')
    cem_controller = CemMPCController(action_space=chosen_env.action_space,
                                      action_dims=1,
                                      dynamics=cartpole_dynamics,
                                      reward_fn=reward_fn,
                                      horizon=50,
                                      samples=20)

    # 4.1 Rollout Once to Test Controller
    print("Rollout Once to Test Controller")
    accumulated_reward = 0
    obs = chosen_env.reset()
    for i in tqdm(range(200)):
        action = int(cem_controller.next_action(obs))
        obs, rew, done, info = chosen_env.step(int(action))
        accumulated_reward += rew
        chosen_env.render()
        if done:
            break
    print(f'Reward: {accumulated_reward}')

    # 4.2 MPC Controller Rollout to Collect Supervised Data
    print("MPC Controller Rollout to Collect Supervised Data")
    remove_data(f'offline/cartpole/{cem_controller.name}')  # clean existing data
    rollout_and_save_data(CustomCartPole,
                          savedir=f'offline/cartpole/{cem_controller.name}',
                          env_configs=[chosen_config],
                          episodes=20,
                          max_steps_per_episode=200,
                          controller=cem_controller)

    # 5.1 Supervised Learning for Good Initialization
    print('------------------------------')
    model_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    ModelCatalog.register_custom_model(model_name, PolicyFCModel)

    model_config = {
        'custom_model': model_name,
        "custom_model_config": {
            'hiddens': [256, 128, 16],
            'offline_dataset': f'offline/cartpole/{cem_controller.name}',
        }
    }

    # 5.2 Supervised RL for Behavior Cloning
    print('------------------------------')
    register_env('MPC-BC-v1', lambda config: CustomCartPole(config))

    rl_config = {
        "train_batch_size": 200,
        'num_workers': 0,
        'env_config': chosen_config,
        'model': model_config,
    }
    results = []
    dqn_trainer = dqn.DQNTrainer(config=rl_config, env='MPC-BC-v1')

    for i in tqdm(range(100)):
        result_mf = dqn_trainer.train_model()
        print(f"\t RL Reward: "
              f"{result_mf['episode_reward_max']:.4f}  |  "
              f"{result_mf['episode_reward_mean']:.4f}  |  "
              f"{result_mf['episode_reward_min']:.4f}")
        results.append(result_mf['episode_reward_mean'])
        if i % 20 == 0:
            checkpoint = dqn_trainer.save()
            print("checkpoint saved at", checkpoint)
        if i % 20 == 0:
            rollout.rollout(dqn_trainer, env_name='CustomCartPole-v1', num_steps=50, num_episodes=1, no_render=False)

    # 6. Goto 3 and Collect Data for Dynamics Prediction and Context


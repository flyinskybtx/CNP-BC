import copy
import glob
import os
import os.path as osp

import numpy as np
from ray.rllib.evaluation import SampleBatchBuilder
from ray.rllib.offline import JsonWriter
from tqdm import tqdm


def remove_data(savedir, pattern='*.json'):
    """ Remove old data according to pattern in 'savedir'

    :param pattern:
    :return:
    """
    savedir = osp.abspath(osp.join(osp.dirname(__file__), savedir))
    old_files = glob.glob(f"{savedir}/{pattern}")
    for f in old_files:
        os.remove(f)
        print(f'Removed file: {f}')


def rollout_and_save_data(env_cls, savedir, env_configs, episodes, max_steps_per_episode, controller=None):
    """Rollout environment to collect runtime data

    :param env_cls:
    :param savedir:
    :param env_configs:
    :param episodes: num_episode per config
    :param max_steps_per_episode:
    :param controller:
    :return:
    """
    savedir = osp.abspath(osp.join(osp.dirname(__file__), savedir))
    batch_builder = SampleBatchBuilder()
    writer = JsonWriter(f"{savedir}")

    pbar = tqdm(total=len(env_configs) * episodes)
    for config in env_configs:
        env = env_cls(config)
        for eps_id in range(episodes):
            obs = env.reset()
            done = False
            t = 0

            while t < max_steps_per_episode:
                if done:
                    break
                if controller is None:
                    action = env.action_space.sample()
                else:
                    action = int(controller.next_action(obs))

                new_obs, rew, done, info = env.step(action)
                batch_builder.add_values(
                    t=t,
                    eps_id=eps_id,
                    obs=obs,
                    new_obs=new_obs,
                    actions=action,
                    dones=done,
                    infos=config,
                )
                obs = new_obs.copy()
                t += 1
            writer.write(batch_builder.build_and_reset())
            pbar.update(1)


def gen_context(env, num_context_points=15, actions=None):
    """

    :param env:
    :param num_context_points:
    :return:
    """
    # ------------------- Generate Context ------------------------- #
    # Get context points
    context_x = []
    context_y = []

    if actions is None:
        actions = [env.action_space.sample() for _ in range(100)]
    else:
        actions = copy.deepcopy(actions)

    while len(actions) > 0:
        obs = env.reset()
        done = False
        while not done and len(actions) > 0:
            action = actions.pop(0)
            new_obs, rew, done, info = env.step(action)
            delta = new_obs - obs
            context_x.append(np.concatenate([obs, np.array([action])]))
            context_y.append(delta)
            obs = new_obs

    idx = np.arange(len(context_x))
    np.random.seed(0)
    np.random.shuffle(idx)
    context_x = np.stack(context_x, axis=0)[idx[:num_context_points]]  # shape: num_points * (state_dim + action_dim)
    context_y = np.stack(context_y, axis=0)[idx[:num_context_points]]  # shape: num_points * state_dim
    return context_x, context_y


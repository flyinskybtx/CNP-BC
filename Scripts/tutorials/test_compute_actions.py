import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env

from Envs.custom_cartpole_v1 import CustomCartPole

if __name__ == '__main__':
    env_name = 'BaselineCartPole-v1'
    env_config = {'masscart': 1.0,
                  'masspole': 0.1,
                  'length': np.random.uniform(low=0.5, high=1.0),
                  'force_mag': 10, }
    register_env(env_name, lambda config: CustomCartPole(config))
    
    env = CustomCartPole(env_config)
    algo = PPOConfig().environment(env=env_name, env_config=env_config).build()

    episode_reward = 0
    terminated = truncated = False
    
    obs, info = env.reset()
    while not terminated and not truncated:
        action = algo.compute_single_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        print(episode_reward)

import random
from typing import Optional, Dict

from gymnasium.envs.classic_control.cartpole import CartPoleEnv


class CustomCartPole(CartPoleEnv):
    def __init__(self, env_config: Dict, **kwargs):
        super().__init__(**kwargs)
        
        # Re-intialize parameters to custom
        self.config = env_config
        self.gravity = 9.8
        self.masscart = env_config.get('masscart') or self.masscart
        self.masspole = env_config.get('masspole') or self.masspole
        self.total_mass = (self.masspole + self.masscart)
        self.length = env_config['length']  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = env_config['force_mag']
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        
        self.max_steps = 200
        self.step_count = None
    
    def step(self, action):
        obs, reward, terminated, _, info = super().step(action)
        if not terminated:
            self.step_count += 1
            self.step_count += 1
            if self.step_count >= self.max_steps:
                terminated = True
        return obs, reward, terminated, False, info
    
    def reset(self, **kwargs):
        obs, info = super(CustomCartPole, self).reset(**kwargs)
        self.step_count = 0
        return obs, info


def make_cartpole_reward(env):
    x_thresh = env.x_threshold
    theta_thresh = env.theta_threshold_radians
    
    def cartpole_reward(state):
        """ Given state(obs) return reward """
        x, x_dot, theta, theta_dot = state
        done = bool(
            x < -x_thresh
            or x > x_thresh
            or theta < -theta_thresh
            or theta > theta_thresh
        )
        if done:
            return 0.0
        else:
            return 1.0
    
    return cartpole_reward


def make_cartpole_reward_on_traj(env: CartPoleEnv):
    x_thresh = env.x_threshold
    theta_thresh = env.theta_threshold_radians
    
    def cartpole_reward_traj(traj):
        reward = 0.0
        for state in traj:
            x, x_dot, theta, theta_dot = state
            done = bool(
                x < -x_thresh
                or x > x_thresh
                or theta < -theta_thresh
                or theta > theta_thresh
            )
            if done:
                break
            else:
                reward += 1.0
        return reward
    
    return cartpole_reward_traj


if __name__ == '__main__':
    config = {
        'masscart': 1.0,
        'masspole': 0.1,
        'length': random.random(),
        'force_mag': 10,
    }
    env = CustomCartPole(config, render_mode='human')
    reward_fn = make_cartpole_reward(env)
    
    env.reset()
    for _ in range(1000):
        env.render()
        obs, rew, done, _, info = env.step(env.action_space.sample())  # take a random action
        print(reward_fn(obs))
        if done:
            break
    env.close()

import random

from gym.envs.classic_control.cartpole import CartPoleEnv


class CustomCartPole(CartPoleEnv):
    def __init__(self, config):
        super().__init__()

        # Re-intialize parameters to custom
        self.gravity = 9.8
        self.masscart = config['masscart']
        self.masspole = config['masspole']
        self.total_mass = (self.masspole + self.masscart)
        self.length = config['length']  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = config['force_mag']
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'


def make_cartpole_reward(env: CartPoleEnv):
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
    env = CustomCartPole(config)
    reward_fn = make_cartpole_reward(env)

    env.reset()
    for _ in range(1000):
        env.render()
        obs, rew, done, info = env.step(env.action_space.sample())  # take a random action
        print(reward_fn(obs))
        if done:
            break
    env.close()

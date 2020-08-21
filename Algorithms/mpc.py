import numpy as np


def cem_optimize(init_mean, reward_func, init_variance=1., samples=20, precision=1e-2,
                 steps=20, nelite=5, constraint_mean=None, constraint_variance=(-999999, 999999)):
    """
        cem_optimize minimizes cost_function by iteratively sampling values around the current mean with a set variance.
        Of the sampled values the mean of the nelite number of samples with the lowest cost is the new mean for the next iteration.
        Convergence is met when either the change of the mean during the last iteration is less then precision.
        Or when the maximum number of steps was taken.
        :param init_mean: initial mean to sample new values around
        :param reward_func: varience used for sampling
        :param init_variance: initial variance
        :param samples: number of samples to take around the mean. Ratio of samples to elites is important.
        :param precision: if the change of mean after an iteration is less than precision convergence is met
        :param steps: number of steps
        :param nelite: number of best samples whose mean will be the mean for the next iteration
        :param constraint_mean: tuple with minimum and maximum mean
        :param constraint_variance: tuple with minumum and maximum variance
        :return: best_traj, best_reward
    """
    mean = init_mean.copy()
    variance = init_variance * np.ones_like(mean)

    step = 1
    diff = 999999

    while diff > precision and step < steps:
        # candidates: (horizon, samples, action_dim)
        candidates = np.stack(
            [np.random.multivariate_normal(m, np.diag(v), size=samples) for m, v in zip(mean, variance)])
        # apply action constraints
        # process continuous random samples into available discrete actions
        candidates = np.clip(np.round(candidates), constraint_mean[0], constraint_mean[1]).astype(np.int)

        rewards = reward_func(candidates)
        sorted_idx = np.argsort(rewards)[::-1]  # descending
        best_reward = rewards[sorted_idx[0]]
        best_traj = candidates[:, sorted_idx[0], :]
        elite = candidates[:, sorted_idx[:nelite], :]  # elite: (horizon, nelite, action_dim)

        new_mean = np.mean(elite, axis=1)
        variance = np.var(elite, axis=1)
        diff = np.abs(np.mean(new_mean - mean))

        # update
        step += 1
        mean = new_mean

    return best_traj, best_reward  # select best to output


class NaiveMPCController:
    def __init__(self, action_space, dynamics, cost_fn, horizon=5, num_simulated_paths=10):
        """

        :param action_space:
        :param dynamics:
        :param cost_fn:
        :param horizon:
        :param num_simulated_paths:
        """
        self.name = 'Naive_MPC'
        self.dynamics = dynamics
        self.horizon = horizon
        self.num_simulated_paths = num_simulated_paths
        self.action_space = action_space
        self.cost_fn = cost_fn

    def get_action(self, state):
        states = np.repeat(np.array(state).reshape(1, -1), self.num_simulated_paths, axis=0)
        trajs = []
        rewards = np.zeros((self.num_simulated_paths, 1))
        for time_idx in range(self.horizon):
            actions = np.array([self.action_space.sample() for _ in range(self.num_simulated_paths)])
            actions = actions.reshape(self.num_simulated_paths, -1)
            trajs.append(actions)

            query_x = np.concatenate([states, actions], axis=-1)
            target_y = self.dynamics.predict(query_x)
            delta_states = target_y['mu']
            states += delta_states

            rewards += np.array([[self.cost_fn(s)] for s in states])  # calculate rewards

        trajs = np.concatenate(trajs, axis=-1)
        best_traj_num = np.argmax(np.sum(rewards, axis=1))
        best_actions = trajs[best_traj_num, :]
        return np.split(best_actions, self.horizon, axis=-1)


class CemMPCController:
    def __init__(self, action_space, action_dims, dynamics, reward_fn, horizon=5, num_simulated_paths=10):
        """

        :param action_space:
        :param action_dims:
        :param dynamics:
        :param reward_fn:
        :param horizon:
        :param num_simulated_paths:
        """
        self.name = 'Cem_MPC'
        self.dynamics = dynamics
        self.horizon = horizon
        self.num_simulated_paths = num_simulated_paths
        self.action_space = action_space
        self.action_dims = action_dims
        self.reward_fn = reward_fn

        self.trajectory = None
        self.state = None

    def next_action(self, obs):
        """

        :param obs:
        :return: single action
        """
        self.state = obs
        if self.trajectory is None:
            # randomly initialize trajectory
            self.trajectory = np.concatenate(
                [np.array([self.action_space.sample()]).reshape(1, -1) for _ in range(self.horizon)])

        self.trajectory, self.expectation = cem_optimize(self.trajectory,
                                                         self._expected_reward,
                                                         constraint_mean=[0, self.action_space.n - 1],
                                                         samples=self.num_simulated_paths)

        # update trajectory for next step
        self.trajectory = np.concatenate([
            self.trajectory[:-1],
            np.array([self.action_space.sample()]).reshape(1, -1)
        ])
        return self.trajectory[0], self.expectation

    def _expected_reward(self, trajs):
        # trajs shape: (horizon, num_samples, action_dims)
        num_samples = trajs.shape[1]
        states = np.repeat(np.expand_dims(self.state, axis=0), num_samples, axis=0)  # shape (num_samples, state_dims)
        history = []
        for action in trajs:
            query_x = np.concatenate([states, action], axis=-1)
            target_y = self.dynamics.predict(query_x)
            states += target_y['mu']
            history.append(states.copy())

        history = np.stack(history, axis=0)
        rewards = [self.reward_fn(history[:, i, :]) for i in range(num_samples)]
        return np.array(rewards)

import copy

import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import *
from mushroom_rl.utils.viewer import Viewer


# Spring Pastels from https://www.heavy.ai/blog/12-color-palettes-for-telling-better-stories-with-your-data
COLOR_PALETTE = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"]
COLOR_PALETTE_RBG = [(253, 127, 111), (126, 176, 213), (178, 224, 97), (189, 126, 190), (255, 181, 90),
                     (255, 238, 101), (190, 185, 219), (253, 204, 229), (139, 211, 199)]


class MovingObsEnv(Environment):
    def __init__(self, time_step=0.01, horizon=1000, gamma=0.995, n_obstacles=8, dx_obs_type='exact',
                 dx_filter_ratio=1., random_target=False, obstacle_vel_scale=1., random_obs=False, return_cost=False):
        self.dt = time_step
        self.n_obs = n_obstacles
        self.boundary = 5
        self.dx_obs_type = dx_obs_type
        self.random_target = random_target
        self.random_obs = random_obs
        self._obstacle_vel_scale = obstacle_vel_scale
        self.return_cost = return_cost
        self.constr_dim = 1

        if not self.random_target:
            observation_space = Box(low=np.array([-self.boundary, -self.boundary] * (1 + self.n_obs) +
                                                 [-3., -3.] * self.n_obs),
                                    high=np.array([self.boundary, self.boundary] * (1 + self.n_obs) +
                                                  [3., 3.] * self.n_obs))
        else:
            observation_space = Box(low=np.array([-self.boundary, -self.boundary] * (1 + self.n_obs) +
                                                 [-3., -3.] * self.n_obs + [-5., -5.]),
                                    high=np.array([self.boundary, self.boundary] * (1 + self.n_obs) +
                                                  [3., 3.] * self.n_obs + [5., 5]))
        action_space = Box(low=-np.ones(2) * 1, high=np.ones(2) * 1)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        self._viewer = Viewer(env_width=11, env_height=11, background=(255, 255, 255))
        self.vel_limits = np.array([3., 3.])

        self._state = np.zeros(2)
        self._obstacle_state = np.zeros((self.n_obs, 2))
        self._obstacle_vel = np.zeros((self.n_obs, 2))
        self._goal_state = np.array([4.5, 4.5])
        self._object_radius = 0.3
        self._obs_vel_prev = np.zeros((self.n_obs, 2))
        self._obs_prev_state = self._obstacle_state.copy()
        self._vel_filter_ratio = dx_filter_ratio

        self.task_info = {
            'n_total_col_obs': 0,
            'n_episode_col_obs': 0,
            'dist_to_obs': np.inf,
            'n_total_episode_col_boundary': 0,
            'n_episode_col_boundary': 0,
            'dist_to_bound': np.inf,
            'dist_to_goal': np.inf
        }
        super().__init__(mdp_info)

    def reset(self, state=None):
        self._state[:2] = np.array([-4.5, -4.5])
        self.t_obs = 0
        if self.random_obs:
            self._obstacle_state = np.random.uniform(-3.5, 3.5, (self.n_obs, 2))
            self._obstacle_vel = np.random.uniform(-1, 1, (self.n_obs, 2))
        else:
            ang = np.linspace(0, 2 * np.pi, self.n_obs + 1)[:-1]
            self._obstacle_state = np.array([np.cos(ang), np.sin(ang)]).T * 3.5
            self._obstacle_vel = np.zeros((self.n_obs, 2))
        self._obs_vel_prev = self._obstacle_vel.copy()
        self._obs_prev_state = self._obstacle_state.copy()

        self.task_info['n_episode_col_obs'] += self.task_info['dist_to_obs'] <= 0
        self.task_info['n_total_col_obs'] += self.task_info['dist_to_obs'] <= 0
        self.task_info['n_episode_col_boundary'] += self.task_info['dist_to_bound'] <= 0
        self.task_info['n_total_episode_col_boundary'] += self.task_info['dist_to_bound'] <= 0
        self.task_info['dist_to_obs'] = np.inf
        self.task_info['dist_to_bound'] = np.inf
        self.task_info['dist_to_goal'] = np.inf

        if self.random_target:
            self._goal_state = np.random.uniform(-4, 4, (2,))
        return self._create_observation()

    def step(self, action):
        action = action[:2]
        action = np.clip(action, self.info.action_space.low, self.info.action_space.high)
        action = action * self.vel_limits

        self._state += action * self.dt
        self.task_info['dist_to_bound'] = np.minimum(self.boundary - np.abs(self._state).max(),
                                                     self.task_info['dist_to_bound'])
        self._state = np.clip(self._state, -5, 5)

        if self.random_obs:
            self._obstacle_state += self._obstacle_vel * self.dt
            self._obstacle_vel += np.random.uniform(-20, 20, self._obstacle_state.shape) * self.dt
            self._obstacle_vel = np.clip(self._obstacle_vel,
                                         -self._obstacle_vel_scale * self.vel_limits[0],
                                         self._obstacle_vel_scale * self.vel_limits[0])
        else:
            self.t_obs += self.dt
            for i in range(self.n_obs):
                dtheta = self.vel_limits[0] * self._obstacle_vel_scale
                radius = 1
                theta_offset = np.pi * 2 / self.n_obs * i
                theta_i = theta_offset + dtheta * self.t_obs
                self._obstacle_state[i] = (radius * np.array([np.cos(theta_i), np.sin(theta_i)]) +
                                           2.5 * np.array([np.cos(theta_offset), np.sin(theta_offset)]))
                # self._obstacle_state[i] += self._obstacle_vel[i] * self.dt
                self._obstacle_vel[i] = dtheta * radius * np.array([-np.sin(theta_i), np.cos(theta_i)])

        clip_idx = np.abs(self._obstacle_state) > 4
        self._obstacle_vel[clip_idx] = -self._obstacle_vel[clip_idx]
        self._obstacle_state = np.clip(self._obstacle_state, -4, 4)

        reward = - np.linalg.norm(self._goal_state - self._state) / 3
        if np.linalg.norm(self._goal_state - self._state) < 0.1:
            reward += 0.1

        dist = np.linalg.norm(self._state - self._obstacle_state, axis=1) - 2 * self._object_radius
        self.task_info['dist_to_obs'] = np.minimum(dist.min(), self.task_info['dist_to_obs'])
        self.task_info['dist_to_goal'] = np.linalg.norm(self._goal_state - self._state)
        obs = self._create_observation()

        if self.task_info['dist_to_goal'] < 0.2 or dist.min() < 0:
            absorb = True
        else:
            absorb = False

        cost = -np.minimum(self.task_info['dist_to_obs'], self.task_info['dist_to_bound'])
        self.task_info['cost'] = cost

        if self.return_cost:
            return obs, reward, cost, absorb, copy.deepcopy(self.task_info)

        return obs, reward, absorb, copy.deepcopy(self.task_info)

    def render(self, record=False):
        offset = np.array([5.5, 5.5])
        self._viewer.circle(center=self._state[:2] + offset,
                            radius=self._object_radius, color=COLOR_PALETTE_RBG[1], width=0)

        for i in range(self.n_obs):
            self._viewer.circle(center=self._obstacle_state[i] + offset, radius=self._object_radius,
                                color=COLOR_PALETTE_RBG[0], width=0)

        self._viewer.square(center=self._goal_state + offset, angle=0, edge=0.6, color=COLOR_PALETTE_RBG[2])

        self._viewer.square(center=offset, angle=0, edge=10, color=(0., 0., 0), width=3)

        self._viewer.display(self.dt * 0.4)

        if record:
            return self._viewer.get_frame()

    def _create_observation(self):
        if self.dx_obs_type == 'exact':
            obs_vel = self._obstacle_vel
            self._obs_prev_state = self._obstacle_state.copy()
        elif self.dx_obs_type == 'None' or self.dx_obs_type is None:
            obs_vel = np.zeros_like(self._obstacle_vel)
            self._obs_prev_state = self._obstacle_state.copy()
        elif self.dx_obs_type == 'filtered':
            obs_state_observation = self._obstacle_state + np.random.randn(*self._obstacle_state.shape) * 0.03
            fd_vel = (obs_state_observation - self._obs_prev_state) / self.dt
            noise_vel = fd_vel + self._obstacle_vel_scale * np.random.randn(*self._obstacle_vel.shape) * 0.0
            obs_vel = (1 - self._vel_filter_ratio) * self._obs_vel_prev + self._vel_filter_ratio * noise_vel
            self._obs_vel_prev = obs_vel.copy()
            self._obs_prev_state = self._obstacle_state.copy()
        else:
            raise ValueError("Unknown type dx_obs_type: ", self.dx_obs_type)

        obs_list = [self._state, self._obstacle_state.flatten(), obs_vel.flatten()]
        if self.random_target:
            obs_list.append(self._goal_state)
        return np.concatenate(obs_list)

    def clear_task_info(self):
        self.task_info['n_episode_col_obs'] = 0
        self.task_info['dist_to_obs']: np.inf
        self.task_info['n_episode_col_boundary'] = 0
        self.task_info['dist_to_bound'] = np.inf
        self.task_info['dist_to_goal'] = np.inf


if __name__ == '__main__':
    env = MovingObsEnv(n_obstacles=7, random_obs=True, return_cost=True)
    env.reset()

    action = np.zeros(2)

    i = 0
    while True:
        env.step(action)
        env.render()
        i += 1
        if i > env.info.horizon:
            env.reset()
            i = 0

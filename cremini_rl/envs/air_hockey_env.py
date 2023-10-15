import torch
import mujoco
import numpy as np
from enum import Enum
from mushroom_rl.utils.spaces import Box
from air_hockey_challenge.environments.iiwas import AirHockeySingle
from air_hockey_challenge.environments.planar.single import AirHockeySingle as PlanarAirHockeySingle
from air_hockey_challenge.environments.position_control_wrapper import PositionControlIIWA
from air_hockey_challenge.constraints import *
from collections import OrderedDict


class AbsorbType(Enum):
    NONE = 0
    GOAL = 1
    UP = 2
    RIGHT = 3
    LEFT = 4
    BOTTOM = 5


class Cache(OrderedDict):
    def __init__(self, maxsize=200000, /, *args, **kwds):
        self.maxsize = maxsize
        super().__init__(*args, **kwds)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]


class AccelerationControl:
    def _compute_action(self, obs, action):
        q, dq = self.get_joints(obs)
        acc_high = np.minimum(self.env_info['robot']['joint_acc_limit'][1],
                              5 * (self.env_info['robot']['joint_vel_limit'][1] - dq))
        acc_low = np.maximum(self.env_info['robot']['joint_acc_limit'][0],
                             5 * (self.env_info['robot']['joint_vel_limit'][0] - dq))
        acc = np.clip(action, acc_low, acc_high)
        self.env_info['robot']['robot_data'].qpos[:] = q
        self.env_info['robot']['robot_data'].qvel[:] = dq
        self.env_info['robot']['robot_data'].qacc[:] = acc
        torque = np.zeros(self.env_info['robot']['n_joints'])
        mujoco.mj_inverse(self.env_info['robot']['robot_model'], self.env_info['robot']['robot_data'])
        torque = self.env_info['robot']['robot_data'].qfrc_inverse
        return torque

    def _modify_mdp_info(self, mdp_info):
        super(AccelerationControl, self)._modify_mdp_info(mdp_info)
        mdp_info.action_space = Box(low=-np.ones(self.env_info['robot']['n_joints']),
                                    high=np.ones(self.env_info['robot']['n_joints']))
        return mdp_info


class AirHockeyEnv(AccelerationControl, AirHockeySingle):
    def __init__(self, return_cost=True):
        self.return_cost = return_cost

        super().__init__(horizon=200)

        # Compute Link Constraint Bound
        x_l = - self.env_info['robot']['base_frame'][0][0, 3] - (
                self.env_info['table']['length'] / 2 - self.env_info['mallet']['radius'])
        y_l = - (self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])
        y_u = self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius']
        z_l = self.env_info['robot']['ee_desired_height'] - 0.02
        z_u = self.env_info['robot']['ee_desired_height'] + 0.02
        z_wr = 0.25
        z_el = 0.25
        self.link_constr_ub = np.array([x_l, y_l, -y_u, z_wr, z_el])
        self.ee_height_ub = np.array([z_l, -z_u])

        self.ee_puck_dist = np.inf
        self.constr_dim = 1

        self._end_effector_const = EndEffectorConstraint(self.env_info)

        self.original_constraint_list = ConstraintList()
        self.original_constraint_list.add(JointPositionConstraint(self.env_info))
        self.original_constraint_list.add(JointVelocityConstraint(self.env_info))
        self.original_constraint_list.add(EndEffectorConstraint(self.env_info))
        self.original_constraint_list.add(LinkConstraint(self.env_info))

    def step(self, action):
        obs, reward, done, info = super(AirHockeyEnv, self).step(action)

        cost = info["cost"]

        if self.return_cost:
            return obs, reward, cost, done, info
        return obs, reward, done, info

    def _compute_ee_height_constraint(self, q):
        q_pos = q[:7]
        q_vel = q[7:]
        # Only take z height endeffector constraints
        cons = self._end_effector_const.fun(q_pos, q_vel)[3:]
        J_q = self._end_effector_const.jacobian(q_pos, q_vel)[3:]

        return cons, J_q

    def setup(self, obs=None):
        puck_pos = np.random.uniform([-0.7, -0.4, -np.pi], [-0.3, 0.4, np.pi])
        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        self._write_data("puck_yaw_pos", puck_pos[2])

        super(AirHockeyEnv, self).setup(obs)

        self.absorb_type = AbsorbType.NONE
        self.ee_puck_dist = np.inf

    def reward(self, state, action, next_state, absorbing):
        puck_pos = next_state[:2].copy()
        puck_pos[0] += 1.51
        puck_vel = next_state[3:5]
        joint_pos = next_state[6:13]

        if absorbing:
            r = 0
            factor = (1 - self.info.gamma ** self.info.horizon) / (1 - self.info.gamma)
            if self.absorb_type == AbsorbType.GOAL:
                r = 1.5 - (np.clip(abs(puck_pos[1]), 0, 0.1) * 5)
            elif self.absorb_type == AbsorbType.UP:
                r = (1 - np.clip(abs(puck_pos[1]) - 0.1, 0, 0.35) * 2)
            elif self.absorb_type == AbsorbType.LEFT:
                r = (0.3 - np.clip(2.43 - puck_pos[0], 0, 1) * 0.3)
            elif self.absorb_type == AbsorbType.RIGHT:
                r = (0.3 - np.clip(2.43 - puck_pos[0], 0, 1) * 0.3)
            r *= factor
        else:
            r = 0

            ee_pos = self.get_ee()[0][:2] + np.array([1.51, 0.])
            ee_puck_dist = np.linalg.norm(ee_pos - puck_pos)
            if self.ee_puck_dist == np.inf:
                self.ee_puck_dist = ee_puck_dist
            elif ee_puck_dist < self.ee_puck_dist:
                r += (self.ee_puck_dist - ee_puck_dist) * 10
                self.ee_puck_dist = ee_puck_dist

            if puck_pos[0] > 1.51:
                r += 0.5 * np.clip(puck_vel[0], 0, 3)

        return r

    def is_absorbing(self, obs):
        puck_pos = obs[:2].copy()
        puck_pos[0] += 1.51
        puck_vel = obs[3:5].copy()

        if puck_pos[0] < 0.58 or (puck_pos[0] < 0.63 and puck_vel[0] > 0.):
            self.absorb_type = AbsorbType.BOTTOM
            return True

        if puck_pos[0] > 2.43 or (puck_pos[0] > 2.39 and puck_vel[0] < 0):
            self.absorb_type = AbsorbType.UP
            if abs(puck_pos[1]) < 0.1:
                self.absorb_type = AbsorbType.GOAL
            return True

        if puck_vel[0] > 0. and puck_pos[0] > 1.51:
            if (puck_pos[1] > 0.45 and puck_vel[1] > 0.) or (puck_pos[1] > 0.42 and puck_vel[1] < 0.):
                self.absorb_type = AbsorbType.LEFT
                return True
            if (puck_pos[1] < -0.45 and puck_vel[0] < 0.) or (puck_pos[1] < -0.42 and puck_vel[1] > 0.):
                self.absorb_type = AbsorbType.RIGHT
                return True
        return False

    def _create_info_dictionary(self, state):
        q, dq = self.get_joints(state)
        ee_pos, ee_vel = self.get_ee()
        q_max = np.concatenate([-q + self.env_info['robot']['joint_pos_limit'][0] * 0.95,
                                q - self.env_info['robot']['joint_pos_limit'][1] * 0.95]).max()
        dq_max = np.concatenate([-dq + self.env_info['robot']['joint_vel_limit'][0] * 0.95,
                                 dq - self.env_info['robot']['joint_vel_limit'][1] * 0.95]).max()

        pos_offset = self.env_info['robot']['base_frame'][0][:3, 3]
        ee_pos = self._data.body("iiwa_1/striker_mallet").xpos - pos_offset
        wr_pos = self._data.body("iiwa_1/link_6").xpos - pos_offset
        el_pos = self._data.body("iiwa_1/link_4").xpos - pos_offset

        link_max = np.array([-ee_pos[0], -ee_pos[1], ee_pos[1], -wr_pos[2], -el_pos[2]]) + self.link_constr_ub

        # ee_height_max = np.array([-ee_pos[2], ee_pos[2]]) + self.ee_height_ub

        cost = max([q_max, dq_max, link_max.max()])
        success = False
        if self.absorb_type == AbsorbType.GOAL:
            success = True

        return {'cost': cost, 'success': success}

    def original_constraint(self, q, x):
        N = len(q)
        q = q.numpy()
        cons = np.zeros((N, 35))
        J_q = np.zeros((N, 35, 14))
        for i in range(N):
            c, J = self._original_constraint(q[i])

            cons[i] = c
            J_q[i] = J

        return torch.from_numpy(cons), torch.from_numpy(J_q), torch.zeros((N, 35, 0))

    def _original_constraint(self, q):
        pos = q[:7]
        vel = q[7:]

        constraint_keys = ["joint_pos_constr", "joint_vel_constr", "ee_constr", "link_constr"]
        constraints = []
        constraints_J = []

        for key in constraint_keys:
            constraints.append(self.original_constraint_list.get(key).fun(pos, vel))
            constraints_J.append(self.original_constraint_list.get(key).jacobian(pos, vel).copy())

        const = np.concatenate(constraints)
        J_q = np.vstack(constraints_J)

        return const, J_q

    def _preprocess_action(self, action):
        return action * self.env_info['robot']['joint_acc_limit'][1]


class PlanarAirhockeyEnv(AccelerationControl, PlanarAirHockeySingle):
    def __init__(self, return_cost=True, dynamic_noise=0):
        self.return_cost = return_cost

        super().__init__(horizon=300)

        # Compute Link Constraint Bound
        x_l = - self.env_info['robot']['base_frame'][0][0, 3] - (
                self.env_info['table']['length'] / 2 - self.env_info['mallet']['radius'])
        y_l = - (self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])
        y_u = self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius']

        self.link_constr_ub = np.array([x_l, y_l, -y_u])

        self.constr_dim = 1

        self.ee_puck_dist = np.inf

        self.original_constraint_list = ConstraintList()
        self.original_constraint_list.add(JointPositionConstraint(self.env_info))
        self.original_constraint_list.add(JointVelocityConstraint(self.env_info))
        self.original_constraint_list.add(EndEffectorConstraint(self.env_info))

        self.dynamic_noise = dynamic_noise
        self.K = np.array([1.0] * 6 + [0.5] * 5 + [0] * 6)

    def step(self, action):
        new_action = action.copy()
        if self.dynamic_noise > 0:
            new_action += np.random.normal(0, self.dynamic_noise, size=action.shape)
            new_action -= self.dynamic_noise * self.get_joints(self._obs.copy())[1]

        obs, reward, done, info = super(PlanarAirhockeyEnv, self).step(new_action)

        cost = info["cost"]

        if self.return_cost:
            return obs, reward, cost, done, info
        return obs, reward, done, info

    def setup(self, obs=None):
        puck_pos = np.random.uniform([-0.7, -0.4, -np.pi], [-0.3, 0.4, np.pi])
        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        self._write_data("puck_yaw_pos", puck_pos[2])

        super(PlanarAirhockeyEnv, self).setup(obs)

        self.absorb_type = AbsorbType.NONE
        self.ee_puck_dist = np.inf
        self.dist_to_goal = np.inf

        # self._debug_mallet_pos = []

    def constraint_func(self, q):
        N = len(q)
        cons = np.zeros((N, 17))
        J_q = np.zeros((N, 17, 6))
        for i in range(N):
            c, J = self._original_constraint(q[i])

            cons[i] = c
            J_q[i] = J

        return cons, J_q, np.zeros((N, 17, 0)), self.K

    def _original_constraint(self, q):
        pos = q[:3]
        vel = q[3:]

        constraint_keys = ["joint_pos_constr", "ee_constr", "joint_vel_constr"]
        constraints = []
        constraints_J = []

        for key in constraint_keys:
            constraints.append(self.original_constraint_list.get(key).fun(pos, vel))
            constraints_J.append(self.original_constraint_list.get(key).jacobian(pos, vel).copy())

        const = np.concatenate(constraints)
        J_q = np.vstack(constraints_J)

        return const, J_q

    def cost(self, obs):
        # obs: [Batch, n]

        q = obs[:, 6:9]
        q_dot = obs[:, 9:12]

        q_max = np.concatenate([-q + self.env_info['robot']['joint_pos_limit'][0] * 0.95,
                                q - self.env_info['robot']['joint_pos_limit'][1] * 0.95], axis=1).max(axis=1)
        dq_max = np.concatenate([-q_dot + self.env_info['robot']['joint_vel_limit'][0] * 0.95,
                                 q_dot - self.env_info['robot']['joint_vel_limit'][1] * 0.95], axis=1).max(axis=1)

        ee_x, ee_y = self.fk(q)

        link_max = np.max(np.stack([-ee_x, -ee_y, ee_y]).T + self.link_constr_ub[None, :], axis=1)

        cost = np.maximum(np.maximum(q_max, dq_max), link_max)

        return cost

    def fk(self, q):
        x = np.cos(q[:, 0]) * 0.55 + np.cos(q[:, 0] + q[:, 1]) * 0.44 + np.cos(q[:, 0] + q[:, 1] + q[:, 2]) * 0.44
        y = np.sin(q[:, 0]) * 0.55 + np.sin(q[:, 0] + q[:, 1]) * 0.44 + np.sin(q[:, 0] + q[:, 1] + q[:, 2]) * 0.44

        # pos_offset = self.env_info['robot']['base_frame'][0][:3, 3]
        return x, y

    def reward(self, state, action, next_state, absorbing):
        # self._debug_mallet_pos.append(self.get_ee()[0][:2].copy())

        puck_pos = next_state[:2].copy()
        puck_pos[0] += 1.51
        puck_vel = next_state[3:5]
        joint_pos = next_state[6:13]

        if absorbing:
            r = 0
            factor = (1 - self.info.gamma ** self.info.horizon) / (1 - self.info.gamma)
            if self.absorb_type == AbsorbType.GOAL:
                r = 1.5 - (np.clip(abs(puck_pos[1]), 0, 0.1) * 5)
            elif self.absorb_type == AbsorbType.UP:
                r = (1 - np.clip(abs(puck_pos[1]) - 0.1, 0, 0.35) * 2)
            elif self.absorb_type == AbsorbType.LEFT:
                r = (0.3 - np.clip(2.43 - puck_pos[0], 0, 1) * 0.3)
            elif self.absorb_type == AbsorbType.RIGHT:
                r = (0.3 - np.clip(2.43 - puck_pos[0], 0, 1) * 0.3)
            r *= factor
        else:
            r = 0

            ee_pos = self.get_ee()[0][:2] + np.array([1.51, 0.])
            ee_puck_dist = np.linalg.norm(ee_pos - puck_pos)
            dist_to_goal = np.linalg.norm(np.array([2.43, 0]) - puck_pos)

            # if self.ee_puck_dist == np.inf:
            #     self.ee_puck_dist = ee_puck_dist
            # elif ee_puck_dist < self.ee_puck_dist:
            #     r += (self.ee_puck_dist - ee_puck_dist) * 50
            #     self.ee_puck_dist = ee_puck_dist

            if self.dist_to_goal == np.inf:
                self.dist_to_goal = dist_to_goal
            # elif dist_to_goal < self.dist_to_goal:
            r += (self.dist_to_goal - dist_to_goal) * 50
            self.dist_to_goal = dist_to_goal
            # if puck_pos[0] > 1.51:
            # r += 2 * np.clip(puck_vel[0], 0, 3)
        return r

    def is_absorbing(self, obs):
        puck_pos = obs[:2].copy()
        puck_pos[0] += 1.51
        puck_vel = obs[3:5].copy()

        if puck_pos[0] < 0.58 or (puck_pos[0] < 0.63 and puck_vel[0] > 0.):
            self.absorb_type = AbsorbType.BOTTOM
            return True

        if puck_pos[0] > 2.43 or (puck_pos[0] > 2.39 and puck_vel[0] < 0):
            self.absorb_type = AbsorbType.UP
            if abs(puck_pos[1]) < 0.1:
                self.absorb_type = AbsorbType.GOAL
            return True

        if puck_vel[0] > 0. and puck_pos[0] > 1.51:
            if (puck_pos[1] > 0.45 and puck_vel[1] > 0.) or (puck_pos[1] > 0.42 and puck_vel[1] < 0.):
                self.absorb_type = AbsorbType.LEFT
                return True
            if (puck_pos[1] < -0.45 and puck_vel[0] < 0.) or (puck_pos[1] < -0.42 and puck_vel[1] > 0.):
                self.absorb_type = AbsorbType.RIGHT
                return True
        return False

    def _create_info_dictionary(self, state):
        q, dq = self.get_joints(state)
        ee_pos, ee_vel = self.get_ee()
        q_max = np.concatenate([-q + self.env_info['robot']['joint_pos_limit'][0] * 0.95,
                                q - self.env_info['robot']['joint_pos_limit'][1] * 0.95]).max()
        dq_max = np.concatenate([-dq + self.env_info['robot']['joint_vel_limit'][0] * 0.95,
                                 dq - self.env_info['robot']['joint_vel_limit'][1] * 0.95]).max()

        pos_offset = self.env_info['robot']['base_frame'][0][:3, 3]
        ee_pos = self._data.body("planar_robot_1/body_ee").xpos - pos_offset

        link_max = np.array([-ee_pos[0], -ee_pos[1], ee_pos[1]]) + self.link_constr_ub

        success = False
        if self.absorb_type == AbsorbType.GOAL:
            success = True

        return {'cost': max(q_max, dq_max, link_max.max()), 'success': success}

    def _preprocess_action(self, action):
        return action * self.env_info['robot']['joint_acc_limit'][1]

    def _modify_observation(self, obs):
        obs = super()._modify_observation(obs)
        new_obs = obs.copy()
        puck_pos, puck_vel = self.get_puck(obs)
        ee_pos, ee_vel = self.get_ee()
        ee_pos[0] += 1.51
        ee_puck_pos = ee_pos[:2] - puck_pos[:2]
        ee_puck_vel = ee_vel[3:5] - puck_vel[:2]

        return np.concatenate([new_obs, ee_puck_pos, ee_puck_vel])

    def _modify_mdp_info(self, mdp_info):
        mdp_info = super()._modify_mdp_info(mdp_info)
        mdp_info.observation_space = Box(high=np.concatenate([mdp_info.observation_space.high, [2, 1, 5, 5]]),
                                         low=np.concatenate([mdp_info.observation_space.high, [-2, -1, 5, 5]]))
        return mdp_info


if __name__ == '__main__':
    env = PlanarAirhockeyEnv()
    env.reset()
    env.render()
    while True:
        action = np.array([0, 0, 1])
        s, r, c, done, info = env.step(action)

        # env.cost(np.tile(s[None, :], (10, 1)))

        env.render()

        if done:
            env.reset()

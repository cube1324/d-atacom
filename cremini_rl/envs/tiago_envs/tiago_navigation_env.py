import os
import numpy as np
from scipy.spatial.transform import Rotation
from mushroom_rl.utils.spaces import Box
from mushroom_rl.utils.angles import shortest_angular_distance
from mushroom_rl.utils.pybullet.observation import PyBulletObservationType

from cremini_rl.envs.tiago_envs.tiago_base_env import TiagoBase, env_dir


class TiagoFetchNavEnv(TiagoBase):
    def __init__(self, gamma=0.995, horizon=500, step_action_function=None, timestep=1 / 30., n_intermediate_steps=1,
                 debug_gui=False, init_state=None, terminate_on_collision=True, save_key_frame=False,
                 is_opponent_moving=True,
                 tiago_file=env_dir + "/models/tiago_urdf/tiago_wheels_with_screen.urdf",
                 kinematic_file=env_dir + "/models/tiago_urdf/tiago_no_wheel_with_screen.urdf",
                 fetch_file=env_dir + "/models/fetch_robot/robots/fetch.urdf"):
        self.fetch_spec = {'target': np.array([0., 0., 0.]), 'file': fetch_file, 'collision_radius': 0.3}

        self.tiago_spec = {'target': np.array([0., 0., 0.]), 'collision_radius': 0.4}

        self.room_range = np.array([[-3.5, -2.5, 0.0], [3.5, 2.5, 0.0]])
        super().__init__(gamma=gamma, horizon=horizon, differential_drive=True, use_head=False, use_torso=False,
                         use_right_arm=False, use_gripper=False, self_collision=False, control='velocity',
                         step_action_function=step_action_function, timestep=timestep,
                         n_intermediate_steps=n_intermediate_steps, debug_gui=debug_gui, init_state=init_state,
                         tiago_file=tiago_file, kinematic_file=kinematic_file)

        self.save_key_frame = save_key_frame
        self.step_counter = 0
        self.count_collide = 0
        self.terminate_on_collision = terminate_on_collision
        self.is_opponent_moving = is_opponent_moving
        self.episode_steps = list()
        self.key_frame_list = list()
        self.final_distance_list = list()

    def add_models(self):
        model_files = dict()
        model_files[self.fetch_spec['file']] = dict(
            useFixedBase=False, basePosition=[2.0, 0.0, 0.], baseOrientation=[0., 0., 1., 0.])

        wall_file = os.path.join(os.path.dirname(__file__), env_dir + "/models/", "wall.urdf")
        model_files[wall_file] = dict(
            useFixedBase=True, basePosition=[0.0, 0.0, 0.0], baseOrientation=[0., 0., 0., 1.])
        return model_files

    def init_post_process(self):
        super(TiagoFetchNavEnv, self).init_post_process()
        self.tiago_spec.update({'id': self.tiago})
        self.fetch_spec['id'] = self._model_map['fetch']
        self.fetch_spec['joint_names'] = ['torso_lift_joint',
                                          'shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint',
                                          'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint',
                                          'wrist_roll_joint',
                                          'r_gripper_finger_joint', 'l_gripper_finger_joint',
                                          'head_pan_joint', 'head_tilt_joint',
                                          'r_wheel_joint', 'l_wheel_joint']

        self.fetch_spec['pb_idx'] = list()
        self.fetch_spec['joint_limits_low'] = list()
        self.fetch_spec['joint_limits_high'] = list()
        self.fetch_spec['joint_velocity_limits'] = list()
        self.fetch_spec['joint_effort_limits'] = list()
        for j_name in self.fetch_spec['joint_names']:
            for j in range(self.client.getNumJoints(self.fetch_spec['id'])):
                joint_info = self.client.getJointInfo(self.fetch_spec['id'], j)
                if joint_info[1].decode('utf-8') == j_name:
                    self.fetch_spec['pb_idx'].append(joint_info[0])
                    self.fetch_spec['joint_limits_low'].append(joint_info[8])
                    self.fetch_spec['joint_limits_high'].append(joint_info[9])
                    self.fetch_spec['joint_effort_limits'].append(joint_info[10])
                    self.fetch_spec['joint_velocity_limits'].append(joint_info[11])

        self.fetch_spec['joint_limits_low'] = np.array(self.fetch_spec['joint_limits_low'])
        self.fetch_spec['joint_limits_high'] = np.array(self.fetch_spec['joint_limits_high'])
        self.fetch_spec['joint_effort_limits'] = np.array(self.fetch_spec['joint_effort_limits'])
        self.fetch_spec['joint_velocity_limits'] = np.array(self.fetch_spec['joint_velocity_limits'])
        self.fetch_spec['wheel_radius'] = 0.0613
        self.fetch_spec['wheel_length'] = 0.372

    def setup(self, state=None):
        fetch_state = np.array([0., 1.57, 1.4, 0., 1.74, 0., 1.57, 0., 0., 0., 0., 0.0])
        for i, state in enumerate(fetch_state):
            self.client.resetJointState(self.fetch_spec['id'], self.fetch_spec['pb_idx'][i], state)
            self.client.setJointMotorControl2(self.fetch_spec['id'], self.fetch_spec['pb_idx'][i],
                                              controlMode=self.client.POSITION_CONTROL,
                                              targetPosition=state)

        tiago_state = np.array([0.1,
                                -1.1, 1.5, 2.8, 1.57, 1.57, -1.57, 0., 0., 0.,
                                -1.1, 1.5, 2.8, 1.57, 1.57, -1.57, 0., 0., 0.,
                                0., 0.])

        tiago_pos = np.random.uniform(*(0.75 * self.room_range))
        tiago_heading = np.random.uniform(-np.pi, np.pi)
        tiago_orientation = Rotation.from_euler('xyz', [0., 0., tiago_heading]).as_quat()
        self._client.resetBasePositionAndOrientation(self.tiago_spec['id'], tiago_pos, tiago_orientation)
        self.tiago_spec['prev_action'] = np.zeros(self.info.action_space.shape)

        success = False
        while not success:
            fetch_pos = np.random.uniform(*(0.75 * self.room_range))
            if np.linalg.norm(fetch_pos - tiago_pos) > 0.8:
                success = True

        fetch_orientation = Rotation.from_euler('xyz', [0., 0., np.random.uniform(-np.pi, np.pi)]).as_quat()
        self.client.resetBasePositionAndOrientation(self.fetch_spec['id'], fetch_pos, fetch_orientation)
        self.fetch_spec['target'] = np.random.uniform(*(0.75 * self.room_range))
        self.client.resetBasePositionAndOrientation(self.fetch_spec['target_id'], self.fetch_spec['target'],
                                                    [0., 0., 0., 1.])

        self.tiago_spec['target'] = np.random.uniform(*(0.75 * self.room_range))
        self.tiago_spec['target_orientation'] = np.random.uniform(-np.pi, np.pi)
        target_ori = Rotation.from_euler('xyz', [0., 0., self.tiago_spec['target_orientation']]).as_quat()
        self.client.resetBasePositionAndOrientation(self.tiago_spec['target_id'], self.tiago_spec['target'],
                                                    target_ori)
        self.step_counter = 0
        self.client.resetDebugVisualizerCamera(4.0, 0, -89.99, (0., 0., 0.))

        periode = 200
        self.idx = [1, 2, 4, 6]
        self.state = np.array([0., 0., 0., 0.])
        self.max_values = np.array([1, 0.3, 0.4, 0.5])
        self.min_values = -self.max_values

        self.v = (self.max_values - self.min_values) / periode
        self.v[0] /= 2
        super(TiagoFetchNavEnv, self).setup(tiago_state)

    def is_absorbing(self, state):
        self.step_counter += 1

        tiago_pose_vel = self._get_2d_base_pose_and_vel(self.tiago_spec['id'])
        tiago_pos = tiago_pose_vel[:2]
        pos_error = self.tiago_spec['target'][:2] - tiago_pos

        if self._in_collision():
            self.count_collide += 1
            if self.save_key_frame:
                self.capture_key_frame()
            if self.terminate_on_collision:
                self.episode_steps.append(self.step_counter)
                self.final_distance_list.append(np.linalg.norm(pos_error))
                return True
        if self.step_counter >= self.info.horizon:
            self.episode_steps.append(self.step_counter)
            self.final_distance_list.append(np.linalg.norm(pos_error))
        return False

    def reward(self, state, action, next_state, absorbing):
        tiago_pose_vel = self._get_2d_base_pose_and_vel(self.tiago_spec['id'])
        tiago_pos = tiago_pose_vel[:2]

        pos_error = self.tiago_spec['target'][:2] - tiago_pos
        heading_error_pos = shortest_angular_distance(tiago_pose_vel[2], np.arctan2(pos_error[1], pos_error[0]))

        pos_error_norm = np.linalg.norm(pos_error)

        sigma = self._sigmoid(20 * (pos_error_norm - 0.3))
        heading_pos = abs(heading_error_pos) * sigma

        action_penalty = np.linalg.norm((action - self.tiago_spec['prev_action']))
        self.tiago_spec['prev_action'] = action.copy()

        reward = -pos_error_norm - heading_pos / np.pi - 0.1 * action_penalty

        if np.linalg.norm(pos_error) < 0.2:
            reward += 10.
        # print(f"reward: {reward}, pos: {-pos_error_norm}, "
        #       f"heading: {- heading_pos / np.pi} action: {-0.1 * action_penalty}")

        if self._in_collision():
            if self.terminate_on_collision:
                reward = -1000
            else:
                reward -= 1

        return reward

    def capture_key_frame(self):
        view_mat = self.client.computeViewMatrixFromYawPitchRoll([0., 0., -0.5], 8, 0, -45, 0., 2)
        proj_mat = self.client.computeProjectionMatrixFOV(70, 1, 0.1, 20)
        img = self.client.getCameraImage(800, 800, view_mat, proj_mat)
        self.key_frame_list.append(img[2])

    def reset_log_info(self):
        self.count_collide = 0
        self.episode_steps = list()
        self.key_frame_list = list()

    def get_log_info(self):
        return self.count_collide, 0, np.mean(self.episode_steps)

    def _modify_mdp_info(self, mdp_info):
        # TODO Modify the order of the observation space
        observation_low = np.array([-3.5, -3.5,
                                    -3.5, -3.5, -1., -1., -1, -1, -np.pi,
                                    -3.5, -3.5, -1., -1., -1, -1, -np.pi,
                                    -1., -1.,
                                    -3.5, -3.5, -1, -1])  # finger pos and vel
        observation_high = -observation_low
        mdp_info.observation_space = Box(observation_low, observation_high)
        mdp_info.action_space.low[self.diff_drive_idx] = -1.0  # The linear velocity limit
        mdp_info.action_space.high[self.diff_drive_idx] = 1.0  # The linear velocity limit
        mdp_info.action_space.low[self.diff_drive_idx + 1] = -np.pi / 3  # The linear velocity limit
        mdp_info.action_space.high[self.diff_drive_idx + 1] = np.pi / 3  # The linear velocity limit
        return mdp_info

    def _custom_load_models(self):
        vis_shape = self.client.createVisualShape(self.client.GEOM_CYLINDER, radius=0.2, length=0.05,
                                                  rgbaColor=[0., 0., 1., 0.4])
        self.fetch_spec['target_id'] = self.client.createMultiBody(baseVisualShapeIndex=vis_shape,
                                                                   basePosition=self.fetch_spec['target'])

        vis_shape = self.client.createVisualShape(self.client.GEOM_CYLINDER, radius=0.2, length=0.05,
                                                  rgbaColor=[1., 0., 0., 0.4])

        self.tiago_spec['target_id'] = self.client.createMultiBody(baseVisualShapeIndex=vis_shape,
                                                                   basePosition=self.tiago_spec['target'])
        return {"fetch_target_id": self.fetch_spec['target_id'], "tiago_target_id": self.tiago_spec['target_id']}

    def _simulation_pre_step(self):
        if self.is_opponent_moving:
            self._move_fetch()

    def _move_fetch(self):
        fetch_pose_vel = self._get_2d_base_pose_and_vel(self.fetch_spec['id'])
        fetch_pos = fetch_pose_vel[:2]
        # fetch_yaw = np.arctan2(fetch_pose_vel[3], fetch_pose_vel[2])
        # fetch_lin_vel = fetch_pose_vel[4:6]
        fetch_yaw = fetch_pose_vel[2]
        fetch_lin_vel = fetch_pose_vel[3:5]

        pos_err = self.fetch_spec['target'][:2] - fetch_pos
        target_angle = np.arctan2(*pos_err[::-1])
        ang_err = shortest_angular_distance(fetch_yaw, target_angle)
        if ang_err > np.pi:
            ang_err -= np.pi * 2
        if ang_err < -np.pi:
            ang_err += np.pi * 2

        if np.linalg.norm(pos_err) < 0.1 and ang_err < 0.1:
            self.fetch_spec['target'] = np.random.uniform(*(0.75 * self.room_range))
            self.client.resetBasePositionAndOrientation(self.fetch_spec['target_id'], self.fetch_spec['target'],
                                                        [0., 0., 0., 1.])

        v = np.clip(0.5 * np.linalg.norm(pos_err), -0.7, 0.7)
        omega = np.clip(1 * ang_err, -1.5, 1.5)

        if (fetch_pos[0] > 2.6 and fetch_lin_vel[0] > 1e-3) or (fetch_pos[0] < -2.6 and fetch_lin_vel[0] < -1e-3) \
                or (fetch_pos[1] > 1.6 and fetch_lin_vel[1] > 1e-3) \
                or (fetch_pos[1] < -1.6 and fetch_lin_vel[1] < -1e-3):
            alpha = np.clip(1 - np.max(np.abs(fetch_pos)[:2] - np.array([2.6, 1.6])) / 0.5, 0., 1.)
            v = alpha * v

        u_left = (v - omega * self.fetch_spec['wheel_length'] / 2.0) / self.fetch_spec['wheel_radius']
        u_right = (v + omega * self.fetch_spec['wheel_length'] / 2.0) / self.fetch_spec['wheel_radius']
        self.client.setJointMotorControl2(self.fetch_spec['id'], self.fetch_spec['pb_idx'][-1],
                                          self.client.VELOCITY_CONTROL, targetVelocity=u_left,
                                          maxVelocity=self.fetch_spec['joint_velocity_limits'][-1])
        self.client.setJointMotorControl2(self.fetch_spec['id'], self.fetch_spec['pb_idx'][-2],
                                          self.client.VELOCITY_CONTROL, targetVelocity=u_right,
                                          maxVelocity=self.fetch_spec['joint_velocity_limits'][-2])

        # ['torso_lift_joint', 'shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint',
        #  'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint', 'r_gripper_finger_joint',
        #  'l_gripper_finger_joint', 'head_pan_joint', 'head_tilt_joint', 'r_wheel_joint', 'l_wheel_joint']

        # idx = 1, 2, 4, 6

        self.v[self.state > self.max_values] *= -1
        self.v[self.state < self.min_values] *= -1

        self.state += self.v

        for i, q in zip(self.idx, self.state):
            self.client.setJointMotorControl2(self.fetch_spec['id'], self.fetch_spec['pb_idx'][i],
                                              controlMode=self.client.POSITION_CONTROL,
                                              targetPosition=q)

    def _create_observation(self, state):
        tiago_pose_vel = self._get_2d_base_pose_and_vel(self.tiago_spec['id'])
        fetch_pose_vel = self._get_2d_base_pose_and_vel(self.fetch_spec['id'])
        tiago_pose_vel = np.concatenate([tiago_pose_vel[:2], [np.cos(tiago_pose_vel[2]), np.sin(tiago_pose_vel[2])],
                                         tiago_pose_vel[3:]])
        fetch_pose_vel = np.concatenate([fetch_pose_vel[:2], [np.cos(fetch_pose_vel[2]), np.sin(fetch_pose_vel[2])],
                                         fetch_pose_vel[3:]])


        finger_state = self.client.getLinkState(self.fetch_spec['id'], 19, computeLinkVelocity=True)
        finger_pos = finger_state[0][:2]
        finger_vel = finger_state[6][:2]

        # obs[0:2] = tiago_target
        # obs[2:9] = tiago_pose_vel
        # obs[9:16] = fetch_pose_vel
        # obs[16:18] = prev_action
        # obs[18:20] = finger_pos
        # obs[20:22] = finger_vel
        observation = np.concatenate([self.tiago_spec['target'][:2], tiago_pose_vel,
                                      fetch_pose_vel, self.tiago_spec['prev_action'], finger_pos, finger_vel])
        return observation

    def _in_collision(self):
        if len(self.client.getContactPoints(self.tiago_spec['id'], self.fetch_spec['id'])) + \
                len(self.client.getContactPoints(self.tiago_spec['id'], self._model_map['wall'])):
            return True
        else:
            return False

    def _get_2d_base_pose_and_vel(self, robot_id):
        pos, ori = self.client.getBasePositionAndOrientation(robot_id)
        angle = Rotation.from_quat(ori).as_euler('xyz')[2]
        lin_vel, rot_vel = self.client.getBaseVelocity(robot_id)
        return np.concatenate([pos[:2], [angle], lin_vel[:2], [rot_vel[2]]])

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    import time
    env = TiagoFetchNavEnv(debug_gui=True)
    env.reset()
    while True:
        time.sleep(0.01)
        action = np.zeros_like(env.info.action_space.high)
        action[1] = 1.
        obs, reward, absorb, _ = env.step(action)
        # print(obs)
        # if absorb or env.step_counter > env.info.horizon:
        #     env.reset()
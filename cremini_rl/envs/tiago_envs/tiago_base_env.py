import os
import numpy as np
import pybullet
import pybullet_data
from mushroom_rl.environments.pybullet import PyBullet, PyBulletObservationType

env_dir = os.path.dirname(__file__)


class TiagoBase(PyBullet):
    def __init__(self, gamma=0.99, horizon=500, differential_drive=True, use_head=False, use_torso=False,
                 use_right_arm=True, use_left_arm=False, use_gripper=False, self_collision=True, K_limit_velocity=0.5,
                 control='velocity', step_action_function=None, timestep=1 / 240., n_intermediate_steps=4,
                 debug_gui=False, init_state=None,
                 tiago_file=env_dir + "/models/tiago_urdf/tiago_wheels_with_screen.urdf",
                 kinematic_file=env_dir + "/models/tiago_urdf/tiago_no_wheel_with_screen.urdf"):
        """
        Constructor.
        Args:
            gamma (float, 0.99): discount factor;
            horizon (int, 500): horizon of the task;
            control(bool, True): If false, the robot in position control mode;
            step_action_function(object, None): A callable function to warp-up the policy action to environment command.
        """
        self.control_flags = {'differential_drive': differential_drive, 'use_head': use_head, 'use_torso': use_torso,
                              'use_right_arm': use_right_arm, 'use_left_arm': use_left_arm, 'use_gripper': use_gripper,
                              'velocity_position': False}
        self.diff_drive_idx = 0
        self.diff_drive_span = 0.4044
        self.wheel_radius = 0.0985
        self.self_collision = self_collision
        self.K_limit_velocity = K_limit_velocity

        if control == 'torque':
            self.control_flags['mode'] = pybullet.TORQUE_CONTROL
        elif control == 'position':
            self.control_flags['mode'] = pybullet.POSITION_CONTROL
        elif control == 'velocity':
            self.control_flags['mode'] = pybullet.VELOCITY_CONTROL
        elif control == 'velocity_position':
            self.control_flags['mode'] = pybullet.POSITION_CONTROL
            self.control_flags['velocity_position'] = True
        else:
            raise NotImplementedError

        self.step_action_function = step_action_function
        pybullet_data_path = pybullet_data.getDataPath()
        self.nq = 21
        self.kinematics_pos = np.zeros(self.nq)
        self.init_state = init_state

        model_files = dict()
        model_files[tiago_file] = dict(useFixedBase=not self.control_flags['differential_drive'],
                                       basePosition=[0., 0., 0.],
                                       baseOrientation=[0., 0., 0., 1.])
        if self_collision:
            model_files[tiago_file].update(dict(flags=pybullet.URDF_USE_SELF_COLLISION))

        actuation_spec, observation_spec = self.construct_act_obs_spec()

        plane_file = os.path.join(pybullet_data_path, "plane.urdf")
        model_files[plane_file] = dict(useFixedBase=True, basePosition=[0.0, 0.0, 0.0],
                                       baseOrientation=[0, 0, 0, 1])

        model_files.update(self.add_models())

        super().__init__(model_files, actuation_spec, observation_spec, gamma,
                         horizon, timestep=timestep, n_intermediate_steps=n_intermediate_steps,
                         debug_gui=debug_gui, size=(500, 500), distance=1.8)

        self.init_post_process()

    def _modify_mdp_info(self, mdp_info):
        if self.control_flags['velocity_position'] or self.control_flags['mode'] == pybullet.VELOCITY_CONTROL:
            for i in range(self.diff_drive_idx):
                joint_info = self.client.getJointInfo(*self._indexer.action_data[i][:2])
                mdp_info.action_space.low[i] = -joint_info[11] * self.K_limit_velocity
                mdp_info.action_space.high[i] = joint_info[11] * self.K_limit_velocity
        return mdp_info

    def add_models(self):
        """
        Add new models into a dictionary

        table_file = os.path.join(pybullet_data_path, "table", "table.urdf")
        {table_file: dict(useFixedBase=True, basePosition=[0.8, 0., 0.],
                          baseOrientation=Rotation.from_euler('xyz', (0, 0, np.pi / 2)).as_quat()}
        """
        return list()

    def init_post_process(self):
        """
        Post-process after loading the model
        """

        self.joint_names = ['torso_lift_joint', 'arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint',
                            'arm_left_4_joint', 'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint',
                            'gripper_left_left_finger_joint', 'gripper_left_right_finger_joint', 'arm_right_1_joint',
                            'arm_right_2_joint', 'arm_right_3_joint', 'arm_right_4_joint', 'arm_right_5_joint',
                            'arm_right_6_joint', 'arm_right_7_joint', 'gripper_right_left_finger_joint',
                            'gripper_right_right_finger_joint', 'head_1_joint', 'head_2_joint']

        self.kinematics_update_idx = list()
        for j, joint_name in enumerate(self.joint_names):
            if joint_name in self._indexer.observation_indices_map.keys():
                self.kinematics_update_idx.append(
                    (j, self._indexer.observation_indices_map[joint_name][PyBulletObservationType.JOINT_POS][0]))

        self.tiago = self._indexer.model_map['tiago_dual']
        if self.control_flags['differential_drive']:
            for j in range(self.client.getNumJoints(self.tiago)):
                if "caster" in self.client.getJointInfo(self.tiago, j)[12].decode('utf-8'):
                    self.client.setJointMotorControl2(self.tiago, j, controlMode=self.client.VELOCITY_CONTROL, force=0)
                    self.client.changeDynamics(self.tiago, j, lateralFriction=0, spinningFriction=0,
                                               rollingFriction=0)
                if "wheel" in self.client.getJointInfo(self.tiago, j)[12].decode('utf-8'):
                    self.client.changeDynamics(self.tiago, j, lateralFriction=1, spinningFriction=0,
                                               rollingFriction=0)

            # Modify the mdp_info the differential drive
            mdp_info = self.info
            mdp_info.action_space.low[self.diff_drive_idx] = -1.0  # The linear velocity limit
            mdp_info.action_space.high[self.diff_drive_idx] = 1.0  # The linear velocity limit
            mdp_info.action_space.low[self.diff_drive_idx + 1] = -np.pi / 3  # The linear velocity limit
            mdp_info.action_space.high[self.diff_drive_idx + 1] = np.pi / 3  # The linear velocity limit
            self._modify_mdp_info(mdp_info)

        if self.self_collision:
            self.collision_mask()

    def collision_mask(self):
        # set all links to a default value: Group: 0000001, Mask: 1111000
        for idx in range(self.client.getNumJoints(self.tiago)):
            self.client.setCollisionFilterGroupMask(self.tiago, idx, collisionFilterGroup=int('00000001', 2),
                                                    collisionFilterMask=int('11111000', 2))

        # set upper left arm links to: Group: 0000010, Mask: 1110100
        for name in ['arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint']:
            self.client.setCollisionFilterGroupMask(*self._indexer.joint_map[name],
                                                    collisionFilterGroup=int('00000010', 2),
                                                    collisionFilterMask=int('11110100', 2))

        # set upper right arm links to: Group: 0000100, Mask: 1101010
        for name in ['arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint', 'arm_right_4_joint']:
            self.client.setCollisionFilterGroupMask(*self._indexer.joint_map[name],
                                                    collisionFilterGroup=int('00000100', 2),
                                                    collisionFilterMask=int('11101010', 2))

        # set lower left arm links to: Group: 0001000, Mask: 1010101
        for name in ['arm_left_4_joint', 'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint']:
            self.client.setCollisionFilterGroupMask(*self._indexer.joint_map[name],
                                                    collisionFilterGroup=int('00001000', 2),
                                                    collisionFilterMask=int('11010101', 2))

        # set lower right arm links to: Group: 0010000, Mask: 0101011
        for name in ['arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint', 'arm_right_7_joint']:
            self.client.setCollisionFilterGroupMask(*self._indexer.joint_map[name],
                                                    collisionFilterGroup=int('00010000', 2),
                                                    collisionFilterMask=int('10101011', 2))

        # set left gripper links to: Group: 0100000, Mask: 1010111
        for name in ['gripper_left_right_finger_joint', 'gripper_left_left_finger_joint']:
            self.client.setCollisionFilterGroupMask(*self._indexer.joint_map[name],
                                                    collisionFilterGroup=int('00100000', 2),
                                                    collisionFilterMask=int('11010111', 2))

        # set right gripper links to: Group: 1000000, Mask: 0101111
        for name in ['gripper_right_right_finger_joint', 'gripper_right_left_finger_joint']:
            self.client.setCollisionFilterGroupMask(*self._indexer.joint_map[name],
                                                    collisionFilterGroup=int('01000000', 2),
                                                    collisionFilterMask=int('10101111', 2))

        # set all gripper links to : Group: 0000000, Mask: 0000000
        for idx in [38, 39, 40, 41, 42, 53, 54, 55, 56, 57]:
            self.client.setCollisionFilterGroupMask(self.tiago, idx, collisionFilterGroup=int('0000000', 2),
                                                    collisionFilterMask=int('00000000', 2))

    def construct_act_obs_spec(self):
        actuation_spec = list()
        observation_pos_spec = list()
        observation_vel_spec = list()

        if self.control_flags['use_torso']:
            actuation_spec.append(('torso_lift_joint', self.control_flags['mode']))
            observation_pos_spec.append(('torso_lift_joint', PyBulletObservationType.JOINT_POS))
            # observation_vel_spec.append(('torso_lift_joint', PyBulletObservationType.JOINT_VEL))
            self.diff_drive_idx += 1

        if self.control_flags['use_left_arm']:
            actuation_spec.append(("arm_left_1_joint", self.control_flags['mode']))
            actuation_spec.append(("arm_left_2_joint", self.control_flags['mode']))
            actuation_spec.append(("arm_left_3_joint", self.control_flags['mode']))
            actuation_spec.append(("arm_left_4_joint", self.control_flags['mode']))
            actuation_spec.append(("arm_left_5_joint", self.control_flags['mode']))
            actuation_spec.append(("arm_left_6_joint", self.control_flags['mode']))
            actuation_spec.append(("arm_left_7_joint", self.control_flags['mode']))
            observation_pos_spec.append(("arm_left_1_joint", PyBulletObservationType.JOINT_POS))
            observation_pos_spec.append(("arm_left_2_joint", PyBulletObservationType.JOINT_POS))
            observation_pos_spec.append(("arm_left_3_joint", PyBulletObservationType.JOINT_POS))
            observation_pos_spec.append(("arm_left_4_joint", PyBulletObservationType.JOINT_POS))
            observation_pos_spec.append(("arm_left_5_joint", PyBulletObservationType.JOINT_POS))
            observation_pos_spec.append(("arm_left_6_joint", PyBulletObservationType.JOINT_POS))
            observation_pos_spec.append(("arm_left_7_joint", PyBulletObservationType.JOINT_POS))
            # observation_vel_spec.append(("arm_left_1_joint", PyBulletObservationType.JOINT_VEL))
            # observation_vel_spec.append(("arm_left_2_joint", PyBulletObservationType.JOINT_VEL))
            # observation_vel_spec.append(("arm_left_3_joint", PyBulletObservationType.JOINT_VEL))
            # observation_vel_spec.append(("arm_left_4_joint", PyBulletObservationType.JOINT_VEL))
            # observation_vel_spec.append(("arm_left_5_joint", PyBulletObservationType.JOINT_VEL))
            # observation_vel_spec.append(("arm_left_6_joint", PyBulletObservationType.JOINT_VEL))
            # observation_vel_spec.append(("arm_left_7_joint", PyBulletObservationType.JOINT_VEL))

            self.diff_drive_idx += 7

            if self.control_flags['use_gripper']:
                actuation_spec.append(("gripper_left_left_finger_joint", pybullet.POSITION_CONTROL))
                actuation_spec.append(("gripper_left_right_finger_joint", pybullet.POSITION_CONTROL))
                observation_pos_spec.append(("gripper_left_left_finger_joint", PyBulletObservationType.JOINT_POS))
                observation_pos_spec.append(("gripper_left_right_finger_joint", PyBulletObservationType.JOINT_POS))
                # observation_vel_spec.append(("gripper_left_left_finger_joint", PyBulletObservationType.JOINT_VEL))
                # observation_vel_spec.append(("gripper_left_right_finger_joint", PyBulletObservationType.JOINT_VEL))
                self.diff_drive_idx += 2

        if self.control_flags['use_right_arm']:
            actuation_spec.append(("arm_right_1_joint", self.control_flags['mode']))
            actuation_spec.append(("arm_right_2_joint", self.control_flags['mode']))
            actuation_spec.append(("arm_right_3_joint", self.control_flags['mode']))
            actuation_spec.append(("arm_right_4_joint", self.control_flags['mode']))
            actuation_spec.append(("arm_right_5_joint", self.control_flags['mode']))
            actuation_spec.append(("arm_right_6_joint", self.control_flags['mode']))
            actuation_spec.append(("arm_right_7_joint", self.control_flags['mode']))
            observation_pos_spec.append(("arm_right_1_joint", PyBulletObservationType.JOINT_POS))
            observation_pos_spec.append(("arm_right_2_joint", PyBulletObservationType.JOINT_POS))
            observation_pos_spec.append(("arm_right_3_joint", PyBulletObservationType.JOINT_POS))
            observation_pos_spec.append(("arm_right_4_joint", PyBulletObservationType.JOINT_POS))
            observation_pos_spec.append(("arm_right_5_joint", PyBulletObservationType.JOINT_POS))
            observation_pos_spec.append(("arm_right_6_joint", PyBulletObservationType.JOINT_POS))
            observation_pos_spec.append(("arm_right_7_joint", PyBulletObservationType.JOINT_POS))
            # observation_vel_spec.append(("arm_right_1_joint", PyBulletObservationType.JOINT_VEL))
            # observation_vel_spec.append(("arm_right_2_joint", PyBulletObservationType.JOINT_VEL))
            # observation_vel_spec.append(("arm_right_3_joint", PyBulletObservationType.JOINT_VEL))
            # observation_vel_spec.append(("arm_right_4_joint", PyBulletObservationType.JOINT_VEL))
            # observation_vel_spec.append(("arm_right_5_joint", PyBulletObservationType.JOINT_VEL))
            # observation_vel_spec.append(("arm_right_6_joint", PyBulletObservationType.JOINT_VEL))
            # observation_vel_spec.append(("arm_right_7_joint", PyBulletObservationType.JOINT_VEL))
            self.diff_drive_idx += 7

            if self.control_flags['use_gripper']:
                actuation_spec.append(("gripper_right_left_finger_joint", pybullet.POSITION_CONTROL))
                actuation_spec.append(("gripper_right_right_finger_joint", pybullet.POSITION_CONTROL))
                observation_pos_spec.append(("gripper_right_left_finger_joint", PyBulletObservationType.JOINT_POS))
                observation_pos_spec.append(("gripper_right_right_finger_joint", PyBulletObservationType.JOINT_POS))
                # observation_vel_spec.append(("gripper_right_left_finger_joint", PyBulletObservationType.JOINT_VEL))
                # observation_vel_spec.append(("gripper_right_right_finger_joint", PyBulletObservationType.JOINT_VEL))
                self.diff_drive_idx += 2

        if self.control_flags['use_head']:
            actuation_spec.append(('head_1_joint', self.control_flags['mode']))
            actuation_spec.append(('head_2_joint', self.control_flags['mode']))
            observation_pos_spec.append(('head_1_joint', PyBulletObservationType.JOINT_POS))
            observation_pos_spec.append(('head_2_joint', PyBulletObservationType.JOINT_POS))
            # observation_vel_spec.append(('head_1_joint', PyBulletObservationType.JOINT_VEL))
            # observation_vel_spec.append(('head_2_joint', PyBulletObservationType.JOINT_VEL))
            self.diff_drive_idx += 2

        if self.control_flags['differential_drive']:
            actuation_spec.append(('wheel_left_joint', pybullet.VELOCITY_CONTROL))
            actuation_spec.append(('wheel_right_joint', pybullet.VELOCITY_CONTROL))
            observation_pos_spec.append(('base_link', PyBulletObservationType.LINK_POS))
            observation_vel_spec.append(('base_link', PyBulletObservationType.LINK_LIN_VEL))
            observation_vel_spec.append(('base_link', PyBulletObservationType.LINK_ANG_VEL))

        return actuation_spec, observation_pos_spec + observation_vel_spec

    def reset(self, state=None):
        observation = super().reset(state)
        return observation

    def setup(self, state=None):
        self.kinematics_pos = np.zeros(self.nq)

        if state is not None:
            for j, joint_name in enumerate(self.joint_names):
                self.kinematics_pos[j] = state[j]
        elif self.init_state is not None:
            for j, joint_name in enumerate(self.joint_names):
                self.kinematics_pos[j] = self.init_state[j]

        for j, joint_name in enumerate(self.joint_names):
            self.client.resetJointState(*self._indexer.joint_map[joint_name], self.kinematics_pos[j])
            self.client.setJointMotorControl2(*self._indexer.joint_map[joint_name],
                                              controlMode=self.client.POSITION_CONTROL,
                                              targetPosition=self.kinematics_pos[j])

    def reward(self, state, action, next_state, absorbing):
        """
        Implement reward
        """
        return 0

    def _compute_action(self, state, action):
        if self.step_action_function is None:
            ctrl_action = action.copy()
        else:
            ctrl_action = self.step_action_function(state, action)

        if self.control_flags['differential_drive']:
            ctrl_action = self.compute_wheel_velocity(action)
        if self.control_flags['velocity_position']:
            ctrl_action = self.joints.positions(self._state) + \
                          action * self._timestep * self._n_intermediate_steps
            ctrl_action = np.clip(ctrl_action, self.joints.limits()[0] + 0.05, self.joints.limits()[1] - 0.05)
        return ctrl_action

    def is_absorbing(self, state):
        """
        Implement Absorbing Condition
        """
        return False

    def compute_wheel_velocity(self, action):
        """
        Convert the [v, omega] to left and right wheel velocity
        """
        ctrl_action = action.copy()
        ctrl_action[self.diff_drive_idx] = (action[self.diff_drive_idx] - action[self.diff_drive_idx + 1] *
                                            self.diff_drive_span / 2.0) / self.wheel_radius
        ctrl_action[self.diff_drive_idx + 1] = (action[self.diff_drive_idx] + action[self.diff_drive_idx + 1] *
                                                self.diff_drive_span / 2.0) / self.wheel_radius
        return ctrl_action

    def get_base_pos_and_ori(self):
        return self.client.getBasePositionAndOrientation(self.tiago)

    def get_joint_states(self):
        result = list()
        for joint in self.joint_names:
            result.append(self.client.getJointState(self.tiago, self._indexer.joint_map[joint][1])[0])
        return result


if __name__ == "__main__":
    env = TiagoBase(differential_drive=True, use_torso=True, use_head=False,
                    use_right_arm=True, use_left_arm=False, use_gripper=True, self_collision=False,
                    control='velocity', debug_gui=True)
    init_state = np.zeros(env.kinematics.pino_model.nq)
    init_state[10] = 1
    env.reset(init_state)
    while True:
        act = np.zeros(env.info.action_space.shape)
        act[-3] = 0.02
        act[-4] = 0.02
        act[-2] = 0.1
        act[-1] = 0.2
        state_, reward_, absorbing_, _ = env.step(act)

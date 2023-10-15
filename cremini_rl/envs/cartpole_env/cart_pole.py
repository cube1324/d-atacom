import os
import copy
import numpy as np
from mushroom_rl.utils.spaces import Box
from mushroom_rl.environments.mujoco import MuJoCo, ObservationType


class CartPoleGoalReaching(MuJoCo):
    def __init__(self, gamma=0.99, horizon=1000, timestep=1 / 240, n_substeps=2, n_intermediate_steps=1,
                 **viewer_params):
        scene = os.path.join(os.path.dirname(__file__), 'cartpole.xml')

        observation_spec = [
            ("slide_pos", "slider", ObservationType.JOINT_POS),
            ("hinge_pos", "hinge_1", ObservationType.JOINT_POS),
            ("slide_vel", "slider", ObservationType.JOINT_VEL),
            ("hinge_vel", "hinge_1", ObservationType.JOINT_VEL),
        ]

        actuation_spec = [
            "slide"
        ]

        self._cart_init_pos = -1

        self.task_info = {'failure_count': 0}

        super().__init__(scene, actuation_spec, observation_spec, gamma, horizon, timestep=timestep,
                         n_substeps=n_substeps, n_intermediate_steps=n_intermediate_steps,
                         additional_data_spec=None, collision_groups=None, max_joint_vel=None, **viewer_params)

    def _modify_mdp_info(self, mdp_info):
        obs_low = np.array([-5, -1, -1, -10, -10])
        obs_high = -obs_low
        mdp_info.observation_space = Box(obs_low, obs_high)
        return mdp_info

    def _modify_observation(self, obs):
        obs = np.array([obs[0], np.sin(obs[1]), np.cos(obs[1]), *obs[2:]])
        return obs

    def setup(self, obs=None):
        self._data.joint("slider").qpos = self._cart_init_pos
        self._data.joint("hinge_1").qpos = 0
        self._data.qvel = np.zeros(2)

    def reward(self, obs, action, next_obs, absorbing):
        if absorbing:
            return 0
        else:
            tip_pos = self._data.site('pole_tip').xpos
            goal_pos = self._data.site('goal').xpos
            reward = np.clip(1 - np.linalg.norm(goal_pos - tip_pos) / 4, 0, 1)
            return reward

    def is_absorbing(self, obs):
        # if np.cos(obs[1]) < 0:
        #     self.task_info['failure_count'] += 1
        #     return True

        # if np.abs(obs[0]) > 10:
        #     return True
        return False

    def _preprocess_action(self, action):
        action = np.clip(action[0], -1, 1)
        return action

    def get_and_reset_task_info(self):
        info = copy.deepcopy(self.task_info)
        self.task_info['failure_count'] = 0
        return info


if __name__ == '__main__':
    env = CartPoleGoalReaching()
    env.reset()
    env.render()
    while True:
        # action = np.array([0.1])
        action = np.random.normal(0.5, 0.1, size=(1,))
        s, r, done, info = env.step(action)
        print(s)
        env.render()
        if done:
            env.reset()

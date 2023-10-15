import numpy as np

from cremini_rl.envs.tiago_envs.tiago_navigation_env import TiagoFetchNavEnv

from mushroom_rl.core import Environment


class TiagoNavigationEnv(Environment):
    def __init__(self, return_cost=True, gamma=0.99, horizon=500, render=False):
        self.base_env = TiagoFetchNavEnv(gamma=gamma, horizon=horizon, debug_gui=render, terminate_on_collision=True)
        self.return_cost = return_cost

        self.constr_dim = 1

        super().__init__(self.base_env.info)

    def step(self, action):
        obs, reward, absorb, info = self.base_env.step(action)

        cost = self._compute_cost(obs)
        info["cost"] = cost

        if self.return_cost:
            return obs, reward, cost, absorb, info
        return obs, reward, absorb, info
    
    def reset(self, state=None):
        return self.base_env.reset(state)

    def render(self, record=False):
        self.base_env.render(record)

    def _compute_cost(self, obs):
        # TODO compute cost with arm
        # obs[0:2] = tiago_target
        # obs[2:9] = tiago_pose_vel
        # obs[9:16] = fetch_pose_vel
        # obs[16:18] = prev_action
        # obs[18:20] = finger_pos
        # obs[20:22] = finger_vel
        pos_tiago = np.array([obs[2], obs[3], np.arctan2(obs[5], obs[4])])
        pos_fetch = np.array([obs[9], obs[10], np.arctan2(obs[12], obs[11])])

        wall_cost = np.array([pos_tiago[0] - self.base_env.room_range[1, 0], pos_tiago[1] - self.base_env.room_range[1, 1],
                         self.base_env.room_range[0, 0] - pos_tiago[0], self.base_env.room_range[0, 1] - pos_tiago[1]]) + \
               self.base_env.tiago_spec['collision_radius'] + 0.1

        fetch_cost = -np.linalg.norm((pos_tiago - pos_fetch)[:2]) + self.base_env.tiago_spec['collision_radius'] + \
               self.base_env.fetch_spec['collision_radius'] + 0.1

        # fetch_finger_cost = -np.linalg.norm(pos_tiago[:2] - np.array([obs[18], obs[19]])) + self.base_env.tiago_spec['collision_radius'] + 0.1
        # IDX = 19, 15, 13, 11
        cost_arm = []
        for i in [11, 13, 15, 19]:
            state = self.base_env.client.getLinkState(self.base_env.fetch_spec['id'], i)
            cost_arm.append(-np.linalg.norm(pos_tiago[:2] - state[0][:2]) + self.base_env.tiago_spec['collision_radius'] + 0.1)

        return np.max(np.append(wall_cost, [fetch_cost] + cost_arm))





if __name__ == "__main__":
    def run_env():

        env = TiagoNavigationEnv(render=True)

        observation = env.reset()
        step = 0
        while True:
            step += 1

            # action = np.random.normal(-.1, 0.5, 2)
            action = np.array([-1, 0])
            observation, reward, cost, done, info = env.step(action)
            print(cost)
            if done or step == env.info.horizon:
                env.reset()
                step = 0

    run_env()
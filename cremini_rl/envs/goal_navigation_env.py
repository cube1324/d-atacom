import numpy as np
import safety_gymnasium

# import to register envs
import cremini_rl.envs.custom_safety_gymnasium

from mushroom_rl.core import Environment, MDPInfo

from mushroom_rl.utils.spaces import Box


class GoalNavigationEnv(Environment):
    def __init__(self, static=False, return_cost=True, horizon=1000, gamma=0.99, render=False):
        self.return_cost = return_cost

        render_mode = "human"

        if static:
            self.base_env = safety_gymnasium.make("StaticGoalNavigation-v0", render_mode=render_mode,
                                                  max_episode_steps=horizon)
        else:
            self.base_env = safety_gymnasium.make("SafetyPointGoal1-v0", render_mode=render_mode,
                                                  max_episode_steps=horizon)

        # self.base_env.unwrapped.task.agent.engine.model.opt.timestep = 0.01

        self.dt = self.base_env.unwrapped.task.agent.engine.model.opt.timestep * 10

        # self.base_env.unwrapped.task.sim_conf.frameskip_binom_n = 2

        action_space = Box(self.base_env.action_space.low, self.base_env.action_space.high)
        obs_space = Box(self.base_env.observation_space.low, self.base_env.observation_space.high)

        mdp_info = MDPInfo(obs_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def reset(self, state=None):
        obs, _ = self.base_env.reset()

        return obs

    def step(self, action):
        observation, reward, cost, terminated, truncated, info = self.base_env.step(action)

        info["cost"] = cost

        if self.return_cost:
            return observation, reward, cost, terminated, info
        return observation, reward, terminated, info

    def render(self, record=False):
        self.base_env.render()


if __name__ == "__main__":
    from cremini_rl.utils.navigation_dynamics import GoalNavigationControlSystem
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    env = GoalNavigationEnv(render=True)

    dyn = GoalNavigationControlSystem()


    def plot_lidar(ax, lidar, center=np.array([0, 0]), offset_theta=0, color="g"):
        thetas = np.pi * 2 * np.arange(0.5, 16, 1) / 16

        # points = np.vstack([np.cos(thetas), np.sin(thetas)]).reshape(16, 2) * thetas.reshape(-1, 1)
        points = [[np.cos(offset_theta + theta) * dist, np.sin(offset_theta + theta) * dist] for theta, dist in
                  zip(thetas, lidar)]

        points = np.array(points) + center
        p_old = Polygon(points, closed=True, fill=False, edgecolor=color)

        ax.add_patch(p_old)

        ax.scatter(*center, c=color)


    def lidar_test():
        obs = env.reset()

        action = np.array([1, 0])

        obs, reward, cost, done, info = env.step(action)

        s = dyn.get_q(obs)

        G = dyn.G(obs.reshape(1, -1), [action[1] < 0])

        f = dyn.f(obs.reshape(1, -1))

        s_dot = f + G @ action.reshape(-1, 1)

        s_new = s + s_dot.reshape(1, -1) * env.dt

        obs, reward, cost, done, info = env.step(action)

        s_new_true = dyn.get_q(obs)

        print(s[0, 3:19])

        print(s_new[0, 3:19])

        print(s_new_true[0, 3:19])

        # fig, ax = plt.subplots()
        #
        # plot_lidar(ax, lidar, pos, 0, "b")
        # plot_lidar(ax, lidar_new, pos_new, theta_new)
        #
        # ax.set_xlim([-1, 1])
        # ax.set_ylim([-1, 1])
        #
        # plt.show()


    # lidar_test()

    def run_env():

        observation = env.reset()
        step = 0
        while True:
            step += 1

            # action = np.random.normal(-.1, 0.5, 2)
            action = np.array([-1, 0])
            observation, reward, cost, done, info = env.step(action)

            agent_obs = observation[:12]
            # print(agent_obs[:3], agent_obs[3:6], agent_obs[6:9])
            print(cost)
            goal_obs = observation[12:28]
            hazards_obs = observation[28:44]
            vases_obs = observation[44:60]

            # print(vases_obs)
            if done or step == env.info.horizon:
                env.reset()
                step = 0


    run_env()

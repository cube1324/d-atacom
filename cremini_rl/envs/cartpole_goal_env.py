import numpy as np

from cremini_rl.envs.cartpole_env.cart_pole import CartPoleGoalReaching

from mushroom_rl.core import Environment


class SafeCartPoleEnv(Environment):
    def __init__(self, return_cost=True, gamma=0.99, horizon=1000):
        self.base_env = CartPoleGoalReaching(gamma=gamma, horizon=horizon)
        self.return_cost = return_cost

        super().__init__(self.base_env.info)

        self.constr_dim = 1

    def step(self, action):
        obs, reward, absorb, info = self.base_env.step(action)

        cost = self._compute_cost(obs)
        info["cost"] = cost

        # print(cost, obs[:2])
        if self.return_cost:
            return obs, reward, cost, absorb, info
        return obs, reward, absorb, info

    def reset(self, state=None):
        return self.base_env.reset(state)

    def render(self, record=False):
        self.base_env.render(record)

    def _compute_cost(self, obs):
        theta = np.arctan2(obs[1], obs[2])

        pole_constraint = np.abs(theta) / 1.5707 - 1
        # wall_constraint = np.abs(obs[0]) / 5 - 1
        # return np.max(np.append(pole_constraint, wall_constraint))
        return pole_constraint


if __name__ == '__main__':
    env = SafeCartPoleEnv(return_cost=False)
    env.reset()
    env.render()
    while True:
        # action = np.array([0.1])
        action = np.random.normal(0, 0.1, size=(1,))
        s, r, done, info = env.step(action)
        # print(c)
        env.render()